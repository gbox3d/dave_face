# Inventory

## 목차

- [Repository](#repository)
- [Top-level structure](#top-level-structure)
- [Entrypoints and key modules](#entrypoints-and-key-modules)
- [Build and validation commands](#build-and-validation-commands)
- [Tests](#tests)
- [Notes](#notes)

## Repository

- Name: `dave_face`
- Path: `/home/goback/works/dave_face`
- Summary: Pi 5 + Hailo-10H 얼굴 인식 데모 모노레포. SCRFD-10G 검출(5점) → 112×112 Umeyama 정렬 → ArcFace MobileFaceNet 512-D → 코사인 유사도. 사진 한 장으로 등록하고, 프레임마다 유사도와 처리 시간(NPU·서버·왕복)을 보여 준다. 인증·보안 기능은 의도적으로 없다.

## Top-level structure

| 경로 | 내용 |
|---|---|
| `apps/api/` | Python 3.13 백엔드 (Flask + gunicorn, HailoRT InferModel API). `pyproject.toml`, `run-web.sh`, `.venv/`(uv, git 제외) |
| `apps/api/dave_face/` | `npu.py` · `scrfd.py` · `pipeline.py` · `db.py` · `web.py` · `__init__.py`(버전) |
| `apps/api/tests/` | unittest (NPU 를 열지 않음) + `faces.py`(가짜 관측 생성) |
| `apps/web/` | React 19 + Vite 8 + TS 7. 엔트리 3개: `index.html`(인식 테스트) · `compare.html`(사진 비교) · `admin.html`(등록) |
| `apps/web/src/` | `Recognize.tsx` · `Compare.tsx` · `Admin.tsx` · `flow.tsx`(공용: 카메라·메뉴·시간·안내) · `api.ts` · `capture.ts` · `Qr.tsx` · `main.tsx`/`compare-main.tsx`/`admin-main.tsx` · `styles.css` |
| `models/5.4.0/` | `SHA256SUMS` 만 추적. HEF 는 `scripts/fetch-hef.sh` 로 받는다 |
| `scripts/` | `setup-venv.sh`(uv) · `fetch-hef.sh` · `bench-npu.py` · `print-qr.py` |
| `deploy/` | `ecosystem.config.cjs`(PM2) · `nginx-dave-face.conf` · `workgate.json` |
| `localfiles/` | git 제외: `dave_face.db`, `admin-token`(0600), `qr/cards.pdf`, `logs/`, `samples/`, `verification/`(실측 기록) |
| `readme.md` | 사람용 사용법·운영 문서 |

## Entrypoints and key modules

- 서버: `apps/api/run-web.sh` → `gunicorn --workers 1 --threads 4 'dave_face.web:create_app()'` (127.0.0.1:`${DAVE_FACE_PORT:-3011}`, `--preload` 금지, `HAILORT_LOGGER_PATH` 기본 `localfiles/logs`).
- `web.py`: `InferenceEngine`(워커 스레드 하나가 SDK·NPU 를 쥐고 프레임을 하나씩 처리, 상태 loading/ready/busy/error/stopped), `create_app(engine, store, token)`.
  - 페이지: `GET /`, `/compare`, `/admin` (끝에 `/` 를 붙이면 404), `/assets/<name>`.
  - API: `GET /api/health`, `/api/version` · `POST /api/recognize`(JPEG → 등록된 모두와 유사도) · `POST /api/embed`(JPEG → 512-D, 저장 없음) · `POST /api/enroll`(관리자, multipart `user_id`+`frame`) · `GET /api/users` · `DELETE /api/users/<id>`.
- `npu.py`: `FaceNPU` = VDevice 1 + InferModel 2 (UINT8 입출력, 호스트 역양자화). `hailo_platform` 을 import 하는 유일한 모듈 — 다른 NPU 로 옮길 때 바꿀 곳은 여기와 모델 파일뿐.
- `pipeline.py`: `FacePipeline.analyze(bgr)` → `Obs`(bbox, kps, eye_dist, roll, yaw, blur, emb, npu_ms) | `Reject("no_face")`. 크기·선명도·자세는 재기만 하고 거르지 않는다.
- `scrfd.py`: letterbox 640, 헤드 (H,C) 매칭, 앵커 디코드, NMS.
- `db.py`: sqlite `users(user_id, created_at, embedding, eye_dist, blur, yaw)` — 사용자당 임베딩 하나, `admin_token()`.
- 웹: `api.ts`(상대 경로 API), `capture.ts`(getUserMedia + JPEG 긴 변 1280 q0.85), `flow.tsx`(useCamera, useHealth, Page 메뉴, Timing, useClearOnRestore).

## Build and validation commands

```bash
bash scripts/setup-venv.sh        # uv venv + hailort 5.4.0 wheel(--no-deps) + gunicorn + segno
bash scripts/fetch-hef.sh         # HEF 받기 + SHA-256 검증
pnpm install && pnpm build        # apps/web/dist
pnpm test                         # tsc --noEmit + vite build + API unittest
env -i HOME=$HOME USER=$USER LANG=$LANG PATH=$NVM_BIN:/usr/bin:/bin pm2 start deploy/ecosystem.config.cjs --only dave-face
pm2 restart dave-face             # API 코드 변경 반영 (웹 dist 변경은 재시작 없이 반영)
curl -s 127.0.0.1:3011/api/health
apps/api/.venv/bin/python scripts/print-qr.py   # 접속 QR 카드 PDF → LAN IPP 프린터
```

## Tests

- `pnpm test` 통과 (2026-10-27): API unittest 44건(`test_scrfd`, `test_pipeline`, `test_db`, `test_engine`, `test_web`, `test_version_sync`) + tsc + vite build. NPU 를 여는 테스트는 없다 (StubEngine·FakeNPU 주입).
- 실제 NPU 확인은 수동: 서비스가 뜬 상태에서 `/api/embed`·`/api/recognize` 에 샘플 JPEG(`localfiles/samples/public`)를 보내거나, `pm2 stop dave-face` 후 `scripts/bench-npu.py`.
- 폰 화면은 자동 테스트가 없다 — iPhone Safari 에서 수동 확인.

## Notes

- Python 의존성은 uv 로만, venv 는 `/usr/bin/python3` + `--system-site-packages`(시스템 cv2·numpy·Pillow·Flask). PATH 의 `python3` 는 다른 venv(3.11)라 쓰지 않는다.
- 웹은 빌드에 호스트 값을 박지 않는다(`base: "./"`, API 는 `new URL("api", document.baseURI)`). 그래서 nginx 접두사 뒤와 :3011 직접 둘 다 동작한다.
- 버전 문자열 4곳은 `test_version_sync` 가 지킨다.
