"""HTTP 계약 테스트 — StubEngine 이 프레임 분석 결과를 정해 준다. NPU 를 열지 않는다."""

from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from dave_face import __version__, web
from dave_face.db import Store
from dave_face.pipeline import Reject
from dave_face.web import MAX_FRAME_BYTES, ApiError, create_app
from tests.faces import GALLERY, frontal, obs, stranger

TOKEN = "test-admin-token"
ADMIN = {"X-Admin-Token": TOKEN}


def jpeg(width=640, height=480):
    return cv2.imencode(".jpg", np.full((height, width, 3), 90, dtype=np.uint8))[1].tobytes()


def png():
    return cv2.imencode(".png", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()


class StubEngine:
    def __init__(self):
        self.state = "ready"
        self.result = frontal()
        self.results = []      # 비어 있지 않으면 차례로 돌려준다
        self.calls = 0

    def status(self):
        return {"state": self.state, "model": web.MODEL}

    def analyze(self, frame):
        assert frame.ndim == 3 and frame.dtype == np.uint8
        self.calls += 1
        result = self.results.pop(0) if self.results else self.result
        if isinstance(result, Exception):
            raise result
        return result


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = StubEngine()
        self.store = Store(Path(self.tmp.name) / "dave_face.db")
        self.client = create_app(self.engine, self.store, TOKEN).test_client()

    def enroll(self, user_id="p1", headers=ADMIN, result=None, frame=True):
        if result is not None:
            self.engine.result = result
        data = {"user_id": user_id}
        if frame:
            data["frame"] = (BytesIO(jpeg()), "frame.jpg")
        return self.client.post("/api/enroll", data=data, headers=headers, content_type="multipart/form-data")

    def recognize(self, body=None, content_type="image/jpeg", result=None):
        if result is not None:
            self.engine.result = result
        return self.client.post("/api/recognize", data=jpeg() if body is None else body, content_type=content_type)

    # --- 상태 ---------------------------------------------------------------
    def test_health_and_version(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json, {"ok": True, "name": "dave-face", "version": __version__,
                                       "state": "ready", "model": web.MODEL})
        self.assertEqual(health.headers["Cache-Control"], "no-store")
        self.assertEqual(health.headers["X-Content-Type-Options"], "nosniff")
        self.engine.state = "loading"
        self.assertEqual(self.client.get("/api/health").status_code, 503)
        self.assertEqual(self.client.get("/api/version").json, {"name": "dave-face", "version": __version__})

    # --- 등록 (사진 한 장) -------------------------------------------------------
    def test_enroll_requires_admin_token(self):
        self.assertEqual(self.enroll(headers={}).status_code, 401)
        self.assertEqual(self.enroll(headers={"X-Admin-Token": "wrong"}).status_code, 401)
        self.assertEqual(self.client.get("/api/users").status_code, 401)
        self.assertEqual(self.client.delete("/api/users/p1").status_code, 401)
        self.assertEqual(self.engine.calls, 0)

    def test_enroll_list_replace_delete(self):
        response = self.enroll()
        self.assertEqual(response.status_code, 200)
        self.assertEqual((response.json["user_id"], response.json["replaced"]), ("p1", False))
        self.assertEqual(set(response.json["face"]), {"bbox", "eye_dist", "yaw", "roll", "blur"})
        self.assertGreater(float(self.store.embeddings()["p1"] @ GALLERY), 0.95)
        self.assertTrue(self.enroll(result=frontal(2)).json["replaced"])  # 같은 ID 는 사진을 바꾼다
        users = self.client.get("/api/users", headers=ADMIN).json
        self.assertEqual([u["user_id"] for u in users], ["p1"])
        self.assertEqual(set(users[0]), {"user_id", "created_at", "eye_dist", "blur", "yaw"})
        self.assertEqual(self.client.delete("/api/users/p1", headers=ADMIN).status_code, 204)
        self.assertEqual(self.client.delete("/api/users/p1", headers=ADMIN).status_code, 404)
        self.assertEqual(self.client.delete("/api/users/Bad%20ID", headers=ADMIN).status_code, 422)

    def test_enroll_needs_a_face(self):
        response = self.enroll(result=Reject("no_face"))
        self.assertEqual((response.status_code, response.json["error"]), (422, "no_face"))
        self.assertEqual(self.store.embeddings(), {})
        self.assertEqual(self.enroll(frame=False).json["error"], "frame_missing")
        self.assertEqual(self.enroll(user_id="Bad ID").status_code, 422)

    # --- 인식 ---------------------------------------------------------------
    def test_recognize_scores_every_user(self):
        reply = self.recognize().json  # 등록 전: 얼굴은 보이지만 비교할 사람이 없다
        self.assertEqual((reply["status"], reply["scores"], reply["match"]), ("ok", [], None))
        self.enroll("p1", result=frontal())
        self.enroll("p2", result=obs(frontal().kps, stranger(5)))
        reply = self.recognize(result=frontal(7)).json
        self.assertEqual(reply["status"], "ok")
        self.assertEqual([s["user_id"] for s in reply["scores"]], ["p1", "p2"])  # 높은 점수부터
        self.assertGreater(reply["scores"][0]["score"], 0.9)
        self.assertLess(reply["scores"][1]["score"], 0.3)
        self.assertEqual((reply["match"], reply["threshold"]), ("p1", web.MATCH_THRESHOLD))
        self.assertIn("elapsed_ms", reply)
        self.assertEqual(reply["npu_ms"], 0.0)  # StubEngine 의 관측은 NPU 시간이 0
        self.assertIsNone(self.recognize(result=obs(frontal().kps, stranger(9))).json["match"])  # 문턱 미달

    def test_recognize_reports_no_face_and_errors(self):
        reply = self.recognize(result=Reject("no_face")).json
        self.assertEqual(reply["status"], "no_face")
        self.assertNotIn("scores", reply)
        calls = self.engine.calls
        self.assertEqual(self.recognize(content_type="application/octet-stream").status_code, 415)
        self.assertEqual(self.recognize(body=b"\xff\xd8" + b"0" * MAX_FRAME_BYTES).status_code, 413)
        self.assertEqual(self.recognize(body=png()).status_code, 422)
        self.assertEqual(self.recognize(body=jpeg(200, 150)).status_code, 422)
        self.assertEqual(self.recognize(body=jpeg(1920, 2)).status_code, 422)
        self.assertEqual(self.recognize(body=b"not an image").status_code, 422)
        self.assertEqual(self.engine.calls, calls)  # 잘못된 프레임은 엔진에 닿지 않는다

    # --- 임베딩 (사진 비교 화면) ---------------------------------------------------
    def embed(self, result=None, content_type="image/jpeg"):
        if result is not None:
            self.engine.result = result
        return self.client.post("/api/embed", data=jpeg(), content_type=content_type)

    def test_embed_returns_the_vector_and_stores_nothing(self):
        a = self.embed(result=frontal(1)).json
        b = self.embed(result=frontal(2)).json
        c = self.embed(result=obs(frontal().kps, stranger(3))).json
        self.assertEqual(a["status"], "ok")
        self.assertEqual(len(a["embedding"]), 512)
        va, vb, vc = (np.array(r["embedding"]) for r in (a, b, c))
        self.assertAlmostEqual(float(np.linalg.norm(va)), 1.0, places=3)
        self.assertGreater(float(va @ vb), 0.9)  # 같은 사람
        self.assertLess(float(va @ vc), 0.3)     # 다른 사람
        self.assertEqual(set(a["face"]), {"bbox", "eye_dist", "yaw", "roll", "blur"})
        self.assertEqual((a["threshold"], a["npu_ms"]), (web.MATCH_THRESHOLD, 0.0))
        self.assertIn("elapsed_ms", a)
        self.assertEqual(self.store.embeddings(), {})  # 저장하지 않는다

    def test_embed_no_face_and_errors(self):
        reply = self.embed(result=Reject("no_face")).json
        self.assertEqual(reply["status"], "no_face")
        self.assertNotIn("embedding", reply)
        self.assertEqual(self.embed(content_type="application/octet-stream").status_code, 415)

    def test_busy_and_unexpected_errors(self):
        self.engine.results = [ApiError(503, {"error": "busy"})]
        response = self.recognize()
        self.assertEqual((response.status_code, response.json["error"]), (503, "busy"))
        self.engine.results = [RuntimeError("secret backend details")]
        with self.assertLogs("dave_face.web", level="ERROR") as logs:
            response = self.recognize()
        self.assertEqual((response.status_code, response.json), (500, {"error": "internal"}))
        self.assertNotIn("secret", response.get_data(as_text=True) + " ".join(logs.output))

    def test_request_size_limit(self):
        response = self.client.post("/api/recognize", data=b"0" * (web.MAX_REQUEST_BYTES + 1),
                                    content_type="image/jpeg")
        self.assertEqual(response.status_code, 413)


class StaticTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dist = Path(tmp.name) / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text('<div id="root"></div>')
        (dist / "admin.html").write_text('<div id="root" data-page="admin"></div>')
        (dist / "compare.html").write_text('<div id="root" data-page="compare"></div>')
        (dist / "assets" / "index-abc.js").write_text("console.log(1)")
        patcher = patch.object(web, "WEB_DIST", dist)
        patcher.start()
        self.addCleanup(patcher.stop)
        store = Store(Path(tmp.name) / "db.sqlite")
        self.client = create_app(StubEngine(), store, TOKEN).test_client()

    def test_pages_and_assets(self):
        with self.client.get("/") as index:
            self.assertEqual(index.status_code, 200)
            self.assertIn(b'id="root"', index.data)
            self.assertEqual(index.headers["Cache-Control"], "no-cache")
        with self.client.get("/admin") as admin:
            self.assertEqual(admin.status_code, 200)
            self.assertIn(b'data-page="admin"', admin.data)
        with self.client.get("/compare") as compare:
            self.assertEqual(compare.status_code, 200)
            self.assertIn(b'data-page="compare"', compare.data)
        with self.client.get("/assets/index-abc.js") as asset:
            self.assertEqual(asset.status_code, 200)

    def test_nothing_else_is_exposed(self):
        for path in ("/admin/", "/compare/", "/assets/../index.html", "/assets/..%2F..%2Fpyproject.toml", "/assets/missing.js",
                     "/localfiles/admin-token", "/models/5.4.0/scrfd_10g.hef", "/dave_face/web.py"):
            with self.subTest(path=path), self.client.get(path) as response:
                self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
