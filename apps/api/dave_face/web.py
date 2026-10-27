"""얼굴 인식 테스트 REST API 와 React 페이지(apps/web/dist — / 테스트, /compare 사진 비교, /admin 관리자) 서빙.

NPU 모델 둘은 워커 스레드 하나가 쥔다. 인증·세션·시도 제한은 없다 — 유사도를 그대로 보여 주는 테스트 도구다.
"""

from __future__ import annotations

import atexit
from concurrent.futures import Future, TimeoutError
from contextlib import contextmanager
import hmac
from io import BytesIO
import logging
from pathlib import Path
from queue import Queue
import re
from threading import Lock, Semaphore, Thread
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from . import __version__
from .db import Store, admin_token
from .pipeline import Reject

LOG = logging.getLogger(__name__)
NAME = "dave-face"
MODEL = "scrfd_10g+arcface_mobilefacenet"
ROOT = Path(__file__).resolve().parents[3]
LOCALFILES = ROOT / "localfiles"
WEB_DIST = ROOT / "apps" / "web" / "dist"
MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_FRAME_BYTES = 1024 * 1024
FRAME_SIDE = (240, 1920)          # JPEG 긴 변 허용 범위 (px). 짧은 변은 최소 120
MATCH_THRESHOLD = 0.5             # 코사인 유사도가 이 이상이면 '일치' — 보정 전 초기값이라 화면에 그대로 보여 준다
_USER_ID = re.compile(r"[a-z0-9_-]{1,32}")


class ApiError(Exception):
    def __init__(self, status: int, payload: dict):
        super().__init__(payload.get("error"))
        self.status, self.payload = status, payload


@contextmanager
def _open_default_session():
    # SDK import 와 장치 초기화는 모두 추론 스레드에서 한다.
    from .npu import FaceNPU
    from .pipeline import FacePipeline
    with FaceNPU() as npu:
        yield FacePipeline(npu)


class InferenceEngine:
    """워커가 모든 SDK 호출을 쥔다. 프레임은 한 번에 하나씩, 다음 요청은 슬롯을 잠깐 기다린다."""

    def __init__(self, *, session_factory=None, timeout=5, wait=1.0):
        self.timeout = timeout
        self.wait = wait
        self._factory = session_factory or _open_default_session
        self._lock = Lock()
        self._slot = Semaphore(1)
        self._state = "loading"
        self._closed = False
        self._jobs = Queue(maxsize=1)
        self._thread = Thread(target=self._work, name="dave-face-inference", daemon=True)
        self._thread.start()

    def status(self):
        with self._lock:
            return {"state": self._state, "model": MODEL}

    def analyze(self, frame):
        """BGR 프레임 -> Obs | Reject."""
        if not self._slot.acquire(timeout=self.wait):
            raise ApiError(503, {"error": "busy"})
        try:
            future = Future()
            with self._lock:
                if self._state != "ready" or self._closed:
                    raise ApiError(503, {"error": "busy" if self._state == "busy" else "unavailable",
                                         "state": self._state})
                self._state = "busy"
                self._jobs.put_nowait((frame, future))
            try:
                return future.result(timeout=self.timeout)
            except TimeoutError as error:
                # 워커는 지금 프레임을 끝낸 뒤에야 다음 것을 받는다.
                raise ApiError(504, {"error": "timeout"}) from error
        finally:
            self._slot.release()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._state = "stopped"
            if self._jobs.empty():
                self._jobs.put_nowait(None)
        self._thread.join(timeout=10)

    def _work(self):
        try:
            with self._factory() as pipeline:
                with self._lock:
                    if self._closed:
                        return
                    self._state = "ready"
                while True:
                    job = self._jobs.get()
                    if job is None:
                        break
                    frame, future = job
                    try:
                        result, failure = pipeline.analyze(frame), None
                    except Exception as error:
                        LOG.error("Face analysis failed (%s)", type(error).__name__)
                        result, failure = None, ApiError(500, {"error": "inference"})
                    with self._lock:
                        self._state = "stopped" if self._closed else "ready"
                        closed = self._closed
                    if failure is not None:
                        future.set_exception(failure)
                    else:
                        future.set_result(result)
                    if closed:
                        break
        except Exception as error:
            LOG.error("NPU initialization/shutdown failed (%s)", type(error).__name__)
            with self._lock:
                self._state = "stopped" if self._closed else "error"


def decode_jpeg(data: bytes):
    """JPEG 바이트 -> BGR 프레임. 형식·크기는 PIL 로 헤더만 먼저 확인한다."""
    if len(data) > MAX_FRAME_BYTES:
        raise ApiError(413, {"error": "frame_too_large", "max_bytes": MAX_FRAME_BYTES})
    try:
        with Image.open(BytesIO(data)) as image:
            if (image.format != "JPEG" or not FRAME_SIDE[0] <= max(image.size) <= FRAME_SIDE[1]
                    or min(image.size) < FRAME_SIDE[0] // 2):
                raise ValueError("not a JPEG within the size range")
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ApiError(422, {"error": "invalid_frame"}) from None
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ApiError(422, {"error": "invalid_frame"})
    return frame


def _user_id(value):
    if not isinstance(value, str) or not _USER_ID.fullmatch(value):
        raise ApiError(422, {"error": "invalid_user_id", "pattern": _USER_ID.pattern})
    return value


def _face(result):
    """Obs -> 화면에 보여 줄 얼굴 값 (보여 주기만 한다)."""
    return {"bbox": [round(float(v)) for v in result.bbox], "eye_dist": round(result.eye_dist, 1),
            "yaw": round(result.yaw, 3), "roll": round(result.roll, 1), "blur": round(result.blur)}


def create_app(engine=None, store=None, token=None):
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    app.json.ensure_ascii = False
    if engine is None:
        engine = InferenceEngine()
        atexit.register(engine.close)
    if store is None or token is None:
        LOCALFILES.mkdir(mode=0o700, exist_ok=True)
    store = store or Store(LOCALFILES / "dave_face.db")
    token = token or admin_token(LOCALFILES / "admin-token")
    app.extensions["inference_engine"] = engine

    def require_admin():
        # 빈 토큰이면 헤더 없는 요청과 같아지므로 무조건 거부한다 (fail closed).
        if not token or not hmac.compare_digest(request.headers.get("X-Admin-Token", "").encode(), token.encode()):
            raise ApiError(401, {"error": "unauthorized"})

    def page(name):
        response = send_from_directory(WEB_DIST, name)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return page("index.html")

    # 접두사 뒤(/dave-face/admin)에서도 ./assets/ 가 맞으려면 끝에 / 가 없어야 한다 — /admin/, /compare/ 는 404.
    @app.get("/admin")
    def admin():
        return page("admin.html")

    @app.get("/compare")
    def compare():
        return page("compare.html")

    @app.get("/assets/<path:name>")
    def assets(name):
        return send_from_directory(WEB_DIST / "assets", name)

    @app.get("/api/health")
    def health():
        status = engine.status()
        ok = status["state"] in ("ready", "busy")
        return jsonify({"ok": ok, "name": NAME, "version": __version__, **status}), 200 if ok else 503

    @app.get("/api/version")
    def version():
        return jsonify({"name": NAME, "version": __version__})

    @app.post("/api/enroll")
    def enroll():
        """관리자. multipart user_id + frame(JPEG 한 장) — 얼굴만 보이면 등록한다. 같은 ID 는 교체."""
        require_admin()
        user_id = _user_id(request.form.get("user_id"))
        file = request.files.get("frame")
        if file is None:
            raise ApiError(422, {"error": "frame_missing"})
        result = engine.analyze(decode_jpeg(file.read(MAX_FRAME_BYTES + 1)))
        if isinstance(result, Reject):
            raise ApiError(422, {"error": result.reason})
        replaced = store.replace_user(user_id, result, time.time())
        return jsonify({"user_id": user_id, "replaced": replaced, "face": _face(result)})

    @app.get("/api/users")
    def users():
        require_admin()
        return jsonify(store.list_users())

    @app.delete("/api/users/<user_id>")
    def delete_user(user_id):
        require_admin()
        if not store.delete_user(_user_id(user_id)):
            raise ApiError(404, {"error": "unknown_user"})
        return "", 204

    def analyze_upload():
        """요청 본문 JPEG 한 장 -> (Obs | Reject, 응답 기본값). 인식·임베딩 엔드포인트가 같이 쓴다."""
        if request.mimetype != "image/jpeg":
            raise ApiError(415, {"error": "use Content-Type: image/jpeg"})
        result = engine.analyze(decode_jpeg(request.get_data()))
        reply = {"threshold": MATCH_THRESHOLD, "npu_ms": round(result.npu_ms, 1)}
        if isinstance(result, Reject):
            reply["status"] = result.reason
        else:
            reply.update({"status": "ok", "face": _face(result)})
        return result, reply

    @app.post("/api/recognize")
    def recognize():
        """프레임 한 장 -> 등록된 모두와의 유사도(높은 순)·문턱·얼굴 값·시간. 점수를 그대로 돌려준다."""
        started = time.monotonic()
        result, reply = analyze_upload()
        if reply["status"] == "ok":
            scores = sorted(({"user_id": uid, "score": round(float(result.emb @ emb), 4)}
                             for uid, emb in store.embeddings().items()), key=lambda s: -s["score"])
            reply.update({"scores": scores,
                          "match": scores[0]["user_id"] if scores and scores[0]["score"] >= MATCH_THRESHOLD else None})
        reply["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return jsonify(reply)

    @app.post("/api/embed")
    def embed():
        """프레임 한 장 -> 임베딩(512-D, L2 정규화)·얼굴 값·시간. 아무것도 저장하지 않는다 —
        사진 비교 화면은 기준 사진의 임베딩을 브라우저에 두고 코사인(두 단위 벡터의 내적)을 직접 잰다."""
        started = time.monotonic()
        result, reply = analyze_upload()
        if reply["status"] == "ok":
            reply["embedding"] = [round(float(v), 5) for v in result.emb]
        reply["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return jsonify(reply)

    @app.errorhandler(ApiError)
    def api_error(error):
        return jsonify(error.payload), error.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify({"error": "request_too_large", "max_bytes": MAX_REQUEST_BYTES}), 413

    @app.errorhandler(Exception)
    def unexpected(error):
        if isinstance(error, HTTPException):
            return error
        LOG.error("Request failed (%s)", type(error).__name__)
        return jsonify({"error": "internal"}), 500

    return app
