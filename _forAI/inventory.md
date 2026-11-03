# Inventory

## 목차

- [Repository](#repository)
- [Top-level structure](#top-level-structure)
- [Entrypoints and key modules](#entrypoints-and-key-modules)
- [Build and validation commands](#build-and-validation-commands)
- [Tests](#tests)
- [Notes](#notes)

## Repository

- Name: `dave_face` (브랜치 `jetson`)
- Path: `/home/vcacore/works/dave_face`
- Installed revision: `jetson`, 2026-11-03 조회 HEAD `26bfcdb` / 포팅 기준 `d4dbdb1`, version `0.1.0`.
- Summary: Jetson Orin Nano 4GB + TensorRT 얼굴 인식 데모 모노레포. SCRFD-10G 검출(5점) → 112×112 Umeyama 정렬 → ArcFace MobileFaceNet 512-D → 코사인 유사도. 모델은 Hailo Model Zoo 의 원본 float ONNX 를 trtexec FP16 엔진으로 만든 것이라 `main`(Pi 5 + Hailo-10H)과 같은 가중치. 사진 한 장으로 등록하고, 프레임마다 유사도와 처리 시간(GPU·서버·왕복)을 보여 준다. 데모 두 화면은 브라우저에서 MediaPipe Face Mesh 를 그린다. 인증·보안 기능은 의도적으로 없다.

## Top-level structure

| 경로 | 내용 |
|---|---|
| `apps/api/` | Python 3.10 백엔드 (Flask + gunicorn, TensorRT 8.6 + cuda-python 12.2). `pyproject.toml`, `run-web.sh`(사전 검사: venv·엔진·GPU 인터락·포트), `.venv/`(uv, git 제외) |
| `apps/api/dave_face/` | `npu.py`(TensorRT 백엔드) · `scrfd.py` · `pipeline.py` · `db.py` · `web.py` · `__init__.py`(버전) |
| `apps/api/tests/` | unittest (GPU 를 열지 않음) + `faces.py`(가짜 관측 생성) |
| `apps/web/` | React 19 + Vite 8 + TS 7. 엔트리 3개: `index.html`(인식 테스트) · `compare.html`(사진 비교) · `admin.html`(등록). `public/assets/mediapipe/`(git 제외: WASM·`face_landmarker.task`) |
| `apps/web/src/` | `Recognize.tsx` · `Compare.tsx` · `Admin.tsx` · `flow.tsx`(공용: 카메라·메뉴·시간·안내·메시 캔버스) · `mesh.ts`(MediaPipe Face Landmarker 로딩·그리기) · `api.ts` · `capture.ts` · `Qr.tsx` · `*-main.tsx` · `styles.css` |
| `models/onnx/` | `SHA256SUMS` 만 추적. `scrfd_10g_bnkps.onnx`·`mbf.onnx`(+zip)는 `scripts/fetch-onnx.sh` 로 받는다 |
| `models/trt/` | git 제외. `scripts/build-engines.sh` 가 만든 `*.fp16.trt<ver>.engine` + `timing.cache` (TRT 버전·GPU 전용) |
| `models/5.4.0/` | Pi 전용 HEF 해시(`main` 과 공유) — 이 브랜치에서는 쓰지 않는다 |
| `scripts/` | `setup-venv.sh` · `fetch-onnx.sh` · `build-engines.sh` · `fetch-mediapipe.sh` · `bench-npu.py` · `print-qr.py` · `fetch-hef.sh`(Pi 전용) |
| `deploy/` | `ecosystem.config.cjs`(PM2) · `nginx-dave-face.conf` · `workgate.json` |
| `localfiles/` | git 제외: `dave_face.db`, `admin-token`(0600), `qr/cards.pdf`, `logs/`(trtexec 빌드 로그·layers.json), `samples/` |
| `readme.md` | 사람용 사용법·운영 문서 (실측 표 포함) |

## Entrypoints and key modules

- 서버: `apps/api/run-web.sh` → `gunicorn --workers 1 --threads 4 'dave_face.web:create_app()'` (127.0.0.1:`${DAVE_FACE_PORT:-3011}`, `--preload` 금지 — CUDA 컨텍스트는 fork 를 건너지 못한다). 기동 전 검사: venv·엔진 파일·`:9082`(oni-detector) 인터락·포트 선점; `umask 077`; `oom_score_adj` 800.
- `web.py`(변경 없음): `InferenceEngine`(워커 스레드 하나가 SDK·GPU 를 쥐고 프레임을 하나씩 처리, 상태 loading/ready/busy/error/stopped), `create_app(engine, store, token)`.
  - 페이지: `GET /`, `/compare`, `/admin` (끝에 `/` 를 붙이면 404), `/assets/<path>`(Vite 빌드 + `mediapipe/` WASM·모델).
  - API: `GET /api/health`, `/api/version` · `POST /api/recognize` · `POST /api/embed` · `POST /api/enroll`(관리자) · `GET /api/users` · `DELETE /api/users/<id>`.
- `npu.py`: `FaceNPU` = `trt.Runtime` 1 + CUDA 스트림 1 + `_Model` 2(엔진·실행 컨텍스트·디바이스 버퍼). 호스트 정규화(SCRFD `(x−127.5)/128`, ArcFace `(x−127.5)/127.5`) → NCHW float32 → `execute_async_v3` → 출력 복사. `detect_raw` 가 `(1,N,C)` → `(n,n,2C)` 로 바꿔 `scrfd._heads` 계약을 지킨다. 로드 때 계약 단언(출력 9개·점수 ∈[0,1]·512-D) + 워밍업. `tensorrt`·`cuda` 를 import 하는 유일한 모듈.
- `pipeline.py`: `FacePipeline.analyze(bgr)` → `Obs`(bbox, kps, eye_dist, roll, yaw, blur, emb, npu_ms) | `Reject("no_face")`.
- `scrfd.py`: letterbox 640, 헤드 (H,C) 매칭, 앵커 디코드, NMS (변경 없음).
- `db.py`: sqlite `users(user_id, created_at, embedding, eye_dist, blur, yaw)`, `admin_token()` (변경 없음).
- 웹: `api.ts`(상대 경로 API, `npu_ms` 키 유지), `capture.ts`, `flow.tsx`(useCamera, useHealth, Page, Timing — 라벨 "GPU", `Stage` 의 `mesh`/`tone` 프롭 + `useFaceMesh`), `mesh.ts`(동적 `import("@mediapipe/tasks-vision")`, `FilesetResolver` → `FaceLandmarker`(GPU→CPU 폴백) → `DrawingUtils` 로 테셀레이션·윤곽·홍채, 실패해도 인식 루프는 계속).

## Build and validation commands

```bash
bash scripts/setup-venv.sh        # uv venv + cuda-python 12.2 + cython + Flask/gunicorn/segno/tomli (시스템 tensorrt·cv2·numpy 공유)
bash scripts/fetch-onnx.sh        # ONNX 받기 + SHA-256 검증 (첫 실행은 해시 출력 후 exit 3 → SHA256SUMS 저장)
systemd-run --user --unit=dave-face-build --collect --same-dir -p OOMPolicy=continue -p MemoryMax=2500M \
    -- nice -n 10 choom -n 1000 bash scripts/build-engines.sh   # 엔진 빌드 (oni-detector 정지, 가용 ≥1.5 GB)
pnpm install --frozen-lockfile && bash scripts/fetch-mediapipe.sh && pnpm build   # apps/web/dist
pnpm test                         # tsc --noEmit + vite build + API unittest
env -i HOME=$HOME USER=$USER LANG=$LANG PATH=/home/vcacore/.nvm/versions/node/v24.22.1/bin:/usr/bin:/bin \
    /home/vcacore/.nvm/versions/node/v24.22.1/bin/pm2 start deploy/ecosystem.config.cjs --only dave-face
curl -s 127.0.0.1:3011/api/health
apps/api/.venv/bin/python scripts/bench-npu.py   # 서비스를 내린 뒤 — 지연·검출·코사인 점검
```

## Tests

- 2026-11-03 인수인계 조회: ONNX/ZIP `SHA256SUMS` 4개 일치, TensorRT FP16 엔진 2개·웹 빌드·가상환경 존재, DB 등록 임베딩 0개, DB·관리자 토큰 권한 0600 확인. Dave는 PM2 목록·저장 목록에 없고 3011은 닫혀 있으며 nginx `/dave-face/`는 502다. Oni 실행을 유지한 조회이므로 새 GPU 성능 측정이나 기동 시험으로 해석하지 않는다.
- `pnpm test` 통과 (2026-10-29): API unittest 44건(`test_scrfd`, `test_pipeline`, `test_db`, `test_engine`, `test_web`, `test_version_sync`) + tsc + vite build. GPU 를 여는 테스트는 없다 (StubEngine·FakeNPU 주입; `test_engine` 이 `tensorrt`/`cuda`/`hailo_platform` 미import 를 단언).
- 실제 GPU 확인은 수동: `scripts/bench-npu.py`(서비스 내린 뒤) 또는 서비스가 뜬 상태에서 `/api/embed`·`/api/recognize` 에 샘플 JPEG(`/usr/share/opencv4/samples/data/lena.jpg`, `localfiles/samples/`).
- 폰 화면(메시 포함)은 자동 테스트가 없다 — iPhone Safari 에서 수동 확인.

## Notes

- 현재 교대 운영과 흥의 첫 실행 절차는 [나노 보드 AI 서비스 인수인계](../../system_info/handover-nano-ai-20261103.md)를 따른다. WorkGate의 `dave-face` 등록은 유지되어 있지만 대기 중에는 헬스가 실패하는 것이 정상이다.
- Python 의존성은 uv 로만, venv 는 `/usr/bin/python3`(3.10) + `--system-site-packages`(시스템 tensorrt 8.6.2·cv2 4.8·numpy 1.21.5·Pillow). `cuda-python==12.2.*` 는 import 때 `cython` 이 필요해 함께 설치한다(numpy 를 끌어오지 않는다).
- 웹은 빌드에 호스트 값을 박지 않는다(`base: "./"`, API 는 `new URL("api", document.baseURI)`, 메시 자산도 `assets/mediapipe/` 상대 경로). nginx 접두사 뒤와 :3011 직접 둘 다 동작한다.
- 버전 문자열 4곳은 `test_version_sync` 가 지킨다 (3.10 에서는 `tomli` 폴백).
- 무거운 단계(엔진 빌드·pnpm·bench)는 `systemd-run --user … -p OOMPolicy=continue -p MemoryMax=…` 임시 유닛에서 돌린다 — 에이전트 셸/VS Code 터널 cgroup 의 OOM 연쇄를 막는다.
