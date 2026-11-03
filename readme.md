# Dave Face Recognition System — Jetson (`jetson` 브랜치)

Jetson Orin Nano(ABP301, 4 GB) + **TensorRT GPU** 얼굴 인식 데모. `main` 브랜치는 Raspberry Pi 5 + Hailo-10H NPU 판이고, 이 브랜치는
가속기 백엔드(`apps/api/dave_face/npu.py`)와 모델 파일·배포 경로만 바꾼 것이다. 사진 한 장으로 얼굴을 등록하고, 카메라 프레임마다
등록된 얼굴들과의 유사도와 처리 시간(GPU·서버·왕복)을 실시간으로 보여 준다. 보드 성능을 보는 용도라 인증·보안 기능은 없다.

- 파이프라인: SCRFD-10G 검출(5점) → 112×112 정렬 → ArcFace MobileFaceNet 512-D → 코사인 유사도.
  모델은 Hailo Model Zoo가 HEF를 컴파일한 **원본 float ONNX**(`scrfd_10g_bnkps.onnx`, `mbf.onnx`) → trtexec FP16 엔진. Pi와 같은 가중치라 NPU ↔ GPU 를 정면 비교할 수 있다.
- `apps/api`: Python 3.10(JetPack 시스템), Flask + gunicorn, TensorRT 8.6 Python API + cuda-python 12.2 — GPU 는 워커 스레드 하나가 쥔다
- `apps/web`: React 19 + Vite 8 — `/` 인식 테스트, `/compare` 사진 비교(데모 두 개), `/admin` 등록(관리자: 등록·삭제·접속 QR). 데모 두 화면은 **MediaPipe Face Mesh** 를 브라우저(폰)에서 돌려 얼굴 위에 메시를 그린다 — 백엔드 부하 없음.
- 운영: PM2 `dave-face` · 포트 3011 · nginx `/dave-face/` · WorkGate 카드 "Dave Face". **GPU 를 `oni-detector` 와 공유**해 둘을 동시에 못 돌린다 — 평상시는 oni-detector 가 돌고 dave-face 는 꺼져 있다([전환 규칙](#gpu-전환-켜기끄기)).
- 보드 전체 현황(oni-detector 와 함께 쓰는 규칙·실측·미결 항목): 바로메모 team-n **#1797**. 문제가 생기면 [문제 해결](#문제-해결) 표부터 본다.

## 현재 상태와 인수인계 (2026-11-03)

**설치 완료·대기 중**이다. 설치 소스는 `jetson` 브랜치 `d4dbdb1`, 버전 `0.1.0`이다. 현재 GPU는 Oni Detector가 사용하며 Dave는 PM2 현재 목록과 재부팅용 저장 목록에 없다. 따라서 3011 연결 실패와 `/dave-face/`의 502는 지금의 대기 상태에서 예상되는 결과다.

흥은 [나노 보드 AI 서비스 인수인계](../system_info/handover-nano-ai-20261103.md)에서 접속 주소 조회·관리자 토큰 위치·Oni/Dave 전환 절차를 먼저 확인하면 된다. 실행·복원 명령은 아래 「실행·배포」에도 있다. Dave가 목록에 있는 동안 `pm2 save`를 하지 않으며 Oni는 삭제하지 않고 정지만 한다.

11월 3일에는 모델 해시·설치 파일·등록 DB(0개 임베딩)·프로세스 상태만 확인했다. 아래 GPU 성능 및 테스트 결과는 10월 29일 기록이며 이번 문서 정리에서 다시 측정한 값이 아니다. 휴대폰 접속용 HTTPS 호스트는 MZOK 장치 `GB_jeton-nano-01`로 현재 값을 조회하고, 토큰이 포함된 QR·URL은 공개 자료에 넣지 않는다.

## 사용법 (폰)

폰 카메라는 **HTTPS 에서만** 열린다 → baro_tunnel 퀵터널 주소로 연다(터널이 재시작하면 주소가 바뀐다).
LAN `http://192.168.0.216/dave-face/` 로 열면 API 는 되지만 카메라가 안 된다. KT 계열 DNS 는 `*.trycloudflare.com` 을 못 풀 수 있다(폰 DNS 를 1.1.1.1 로).

**접속 QR 카드** — 테스트 카드와 관리자 카드(토큰이 들어 있어 자동 입력)를 뽑아 폰으로 찍는다:

```bash
cd ~/works/dave_face
apps/api/.venv/bin/python scripts/print-qr.py              # 터널 주소·토큰 → localfiles/qr/cards.pdf → LAN IPP 프린터
apps/api/.venv/bin/python scripts/print-qr.py --no-print   # PDF 만
```

수동으로: 터널 주소 `grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' ~/works/baro_tunnel/localfiles/keeper-out.log | tail -1`
뒤에 `/dave-face/`(테스트) 또는 `/dave-face/admin`(관리자 — 끝에 `/` 없이). 토큰은 `cat localfiles/admin-token`(첫 실행 때 생성).
이 보드의 DB 는 비어 있으니 `/admin` 에서 다시 등록한다(Pi 와 같은 가중치지만 INT8 ↔ FP16 수치가 조금 다르다).

1. **등록 (관리자)** — ID 입력 → **카메라 켜서 등록** → 얼굴이 잡히면 **촬영해 등록**. 사진 한 장. 같은 ID 는 사진이 바뀐다.
2. **테스트** — **인식 시작** → 프레임마다 등록된 모두와의 유사도 막대(높은 순), 문턱 이상이면 "일치 · id", 얼굴 값, 시간(fps · 서버 · GPU · 왕복), 얼굴 위 메시(일치 초록 · 불일치 빨강). **정지**로 끝낸다.
3. **사진 비교** (`/dave-face/compare`, 등록·토큰 없음) — **카메라 켜기** → **기준 사진 찍기** → 카메라 얼굴과 기준 사진의 유사도를 실시간으로 본다. 아무것도 저장하지 않는다.

## 수치 보는 법

| 값 | 뜻 |
|---|---|
| 유사도 | ArcFace 임베딩의 코사인 (−1~1). 문턱 0.5 이상이면 "일치". 보정 전 값(`apps/api/dave_face/web.py` `MATCH_THRESHOLD`) |
| GPU | SCRFD + ArcFace 두 번의 가속기 호출 시간(`npu_ms`) — 호스트 정규화·복사·GPU 실행·동기화 포함 |
| 서버 | 요청을 받아 답하기까지 (JPEG 디코드 + GPU + 비교) |
| 왕복 | 폰이 잰 요청~응답 (터널·네트워크 포함) |
| 눈 거리 · yaw · roll · 선명도 | 얼굴 크기(px) · 좌우 돌림 비율 · 기울기(°) · Laplacian 분산. 보여 주기만 하고 거르지 않는다 |

실측 (2026-10-29, 전원 10W = Orin Nano 4GB·JetPack 6.0 의 최대 모드):

| | Pi 5 + Hailo-10H (main) | Jetson Orin Nano 4GB (jetson) |
|---|---|---|
| SCRFD-10G 640² | NPU 41 ms | GPU compute 14.8 ms · Python `infer` 중앙값 30.7 ms |
| ArcFace MobileFaceNet | NPU 6.7 ms | GPU compute 2.0 ms · Python `infer` 중앙값 3.5 ms |
| `/api/recognize` 서버 중앙값(localhost 20회) | 67 ms (nginx 경유) | 58 ms (`npu_ms` 41.8 ms) |
| 워커 메모리 | ≈200 MB | RSS 462 MB (+ GPU 메모리는 RSS 에 안 잡힘) |
| 엔진 빌드 | — (HEF 다운로드) | mbf 78 s · scrfd 192 s (trtexec, 1회) |

Jetson 의 GPU 시간에는 Pi 에서 HEF 안에 있던 정규화(numpy, 5~10 ms)가 들어간다. 같은 사진끼리 cos 1.000, 다른 사람 ≈0.

## 저장소 구조

```
apps/api/dave_face/   npu.py       TensorRT 런타임 + 엔진 2개 (tensorrt·cuda 를 import 하는 유일한 모듈)
                      scrfd.py     letterbox, 헤드 (H,C) 매칭, 앵커 디코드, NMS
                      pipeline.py  Umeyama 정렬, 선명도·자세 값, FacePipeline.analyze → Obs | Reject("no_face")
                      db.py        sqlite3 users(사용자당 임베딩 하나), 관리자 토큰
                      web.py       InferenceEngine + REST + 페이지 서빙 (/ 테스트, /compare 사진 비교, /admin 등록)
apps/api/tests/       unittest — GPU 를 열지 않는다 (가짜 엔진 주입)
apps/web/src/         Recognize.tsx · Compare.tsx · Admin.tsx · Qr.tsx · flow.tsx(공용·메뉴·시간·메시 캔버스) · mesh.ts(MediaPipe) · api.ts · capture.ts
apps/web/public/assets/mediapipe/   WASM + face_landmarker.task (git 제외, scripts/fetch-mediapipe.sh)
models/onnx/          Hailo Zoo 원본 ONNX (git 제외) + SHA256SUMS
models/trt/           이 보드에서 만든 엔진 *.fp16.trt8.6.2.engine (git 제외, TRT 버전·GPU 전용)
models/5.4.0/         Pi 전용 HEF 해시 (main 브랜치용 — 여기서는 쓰지 않는다)
scripts/              setup-venv.sh · fetch-onnx.sh · build-engines.sh · fetch-mediapipe.sh · bench-npu.py · print-qr.py · fetch-hef.sh(Pi 전용)
deploy/               ecosystem.config.cjs (PM2) · nginx-dave-face.conf · workgate.json
localfiles/           git 제외: DB, admin-token, qr/cards.pdf, logs/(빌드 로그 포함), samples/
```

## 설치 (새 클론)

JetPack 6.0(L4T 36.3, CUDA 12.2, TensorRT 8.6.2, Python 3.10) 기준. Python 의존성은 uv 로만, 시스템 패키지(tensorrt·cv2·numpy)는 그대로 쓴다.

```bash
bash scripts/setup-venv.sh      # uv venv(/usr/bin/python3, --system-site-packages) + cuda-python 12.2 + Flask/gunicorn/segno/tomli
bash scripts/fetch-onnx.sh      # Hailo Zoo ONNX 두 개 다운로드 + SHA-256 검증 (models/onnx)
pm2 stop oni-detector           # GPU 메모리 확보 (4 GB 보드 — 빌드는 가용 메모리 1.5 GB 이상 필요)
systemd-run --user --unit=dave-face-build --collect --same-dir -p OOMPolicy=continue -p MemoryMax=2500M \
    -- nice -n 10 choom -n 1000 bash scripts/build-engines.sh     # models/trt 엔진 (수 분~수십 분, 1회). 로그 localfiles/logs/
pnpm install --frozen-lockfile && bash scripts/fetch-mediapipe.sh && pnpm build   # apps/web/dist (+ WASM·메시 모델 복사)
```

무거운 단계(엔진 빌드·pnpm)는 위처럼 `systemd-run --user` 임시 유닛으로 격리해 돌린다 — 다른 cgroup 의 OOM 연쇄를 막는다.

## 실행·배포

### GPU 전환 (켜기·끄기)

4 GB 를 CPU·GPU 가 나눠 쓴다. `oni-detector`(oni_detector 1.5.0, torch)는 혼자 돌 때도 GPU(nvmap) 1.4 GB + 스왑 1.2 GB 를 쥐고
보드 가용 메모리가 ≈0.3 GB 만 남는다(2026-11-03 실측). 그래서 dave-face 와 **한 번에 하나만** 돈다. `run-web.sh` 는 `:9082` 가 열려 있으면 기동을 거부한다(exit 2).

```bash
cd ~/works/dave_face
pm2 list; ss -H -ltn '( sport = :9082 or sport = :3011 )'   # 지금 누가 GPU 를 쥐었나 (9082 = oni, 3011 = dave-face)

# 얼굴 인식 데모 켜기
pm2 stop oni-detector
env -i HOME=$HOME USER=$USER LANG=$LANG PATH=/home/vcacore/.nvm/versions/node/v24.22.1/bin:/usr/bin:/bin \
    /home/vcacore/.nvm/versions/node/v24.22.1/bin/pm2 start deploy/ecosystem.config.cjs --only dave-face
curl -s 127.0.0.1:3011/api/health          # "state": "ready" 까지 2~3 s

# 끄기 = 평상시 상태로
pm2 delete dave-face && pm2 start oni-detector
```

- `pm2 start` 는 호출한 셸의 환경변수를 앱에 복사한다 → 위처럼 `env -i` 와 pm2 절대 경로로 띄운다.
- `pm2 save` 는 dave-face 가 목록에 **없을 때만** 한다. PM2 7.1.1 은 저장된 앱을 상태와 무관하게 부팅 때 띄우므로 두 GPU 앱이 같이 떠 서로 OOM 을 낸다.
- `pm2 delete oni-detector` 금지(다른 팀 등록이다 — stop 만). 끄기는 `pm2 stop` 이 아니라 `pm2 delete dave-face`(목록에서 지워야 저장 사고가 없다).
- WorkGate 의 Dave Face 행은 꺼져 있는 동안 빨간 것이 정상이다.

### 그 밖의 명령

```bash
pm2 restart dave-face          # 코드·빌드 변경 반영
pm2 logs dave-face
apps/api/run-web.sh            # 개발용 직접 실행 (127.0.0.1:3011, 위 사전 검사 동일)
```

DB 스키마를 바꾸면 `localfiles/dave_face.db` 를 지우고 재시작한다 (마이그레이션 없음). nginx 는 `/etc/nginx/sites-available/default` 의
`include /home/vcacore/works/dave_face/deploy/nginx-dave-face.conf;` 한 줄 — 저장소를 옮기거나 지우기 전에 그 줄을 먼저 빼고 `sudo nginx -t && sudo systemctl reload nginx`.
WorkGate 갱신: `node ~/works/WorkGate/scripts/py.cjs registry put ./deploy/workgate.json --by <작업자>`.

## 테스트·점검

```bash
pnpm test                      # tsc + vite build + API unittest (GPU 불필요)
curl -s 127.0.0.1:3011/api/health
curl -s -H 'content-type: image/jpeg' --data-binary @face.jpg 127.0.0.1:3011/api/recognize   # 사진 한 장의 유사도

pm2 stop dave-face             # GPU 직접 점검은 서비스를 먼저 내린다 (엔진·CUDA 컨텍스트가 두 벌 올라가면 안 된다)
apps/api/.venv/bin/python scripts/bench-npu.py                          # 입출력 shape, 지연시간(중앙값·p95)
apps/api/.venv/bin/python scripts/bench-npu.py --image a.jpg --draw out.jpg   # 검출 결과 그리기
apps/api/.venv/bin/python scripts/bench-npu.py --pair a.jpg b.jpg       # 두 얼굴 코사인
/usr/src/tensorrt/bin/trtexec --loadEngine=models/trt/scrfd_10g.fp16.trt8.6.2.engine --warmUp=500 --iterations=50   # GPU 만의 시간
pm2 start dave-face
```

## REST API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/`, `/compare`, `/admin` | 인식 테스트, 사진 비교, 등록(관리자) 페이지 (끝에 `/` 를 붙이면 404) |
| GET | `/api/health`, `/api/version` | 상태 (ready/busy 200, 그 외 503), 버전 |
| POST | `/api/recognize` | 본문 = JPEG → `{status, threshold, face?, scores?, match?, elapsed_ms, npu_ms}` (`npu_ms` = GPU 시간; 키 이름은 Pi 와 호환) |
| POST | `/api/embed` | 본문 = JPEG → `{status, threshold, face?, embedding?(512), elapsed_ms, npu_ms}` — 저장하지 않는다 |
| POST | `/api/enroll` | 관리자(`X-Admin-Token`). multipart `user_id` + `frame`(JPEG 한 장) → `{user_id, replaced, face}` · 얼굴 없음 422 |
| GET | `/api/users` | 관리자. 등록 목록 |
| DELETE | `/api/users/<id>` | 관리자. 삭제 |
| GET | `/assets/mediapipe/...` | 메시용 WASM·모델 (같은 출처에서만 받는다 — 외부 요청 없음) |

## 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| `/dave-face/` 가 502 | dave-face 가 꺼져 있다(평상시 상태) | [켜기](#gpu-전환-켜기끄기) |
| `pm2 logs` 에 `oni-detector holds the GPU (:9082 listening)` 후 errored | oni-detector 가 돌고 있다 | `pm2 stop oni-detector` 후 `pm2 restart dave-face` |
| `엔진이 없습니다: models/trt/...engine` | 새 클론, 또는 JetPack/TensorRT 가 바뀌어 파일명의 버전이 다르다 | `scripts/fetch-onnx.sh` → `scripts/build-engines.sh`(oni 정지, 가용 ≥1.5 GB) |
| `127.0.0.1:3011 이 이미 열려 있습니다` | dave-face 가 이미 떠 있다(pm2 와 직접 실행 중복) | `pm2 list`, `ss -ltnp 'sport = :3011'` |
| health 가 `503`, `"state": "error"` 로 굳음 | 엔진 로드 실패·GPU 메모리 부족 | `pm2 logs dave-face` 확인 후 `pm2 restart dave-face` |
| 폰에서 주소가 안 열린다 | 퀵터널 주소는 `baro-tunnel-auto` 가 재시작할 때마다 바뀐다(pm2 데몬 재시작·재부팅 포함) | keeper 로그에서 새 주소를 다시 읽고 QR 을 새로 뽑는다(아래). 터널을 재시작하지 말 것 — 주소가 또 바뀐다 |
| 새 주소도 폰 데이터망에서만 안 열린다 | KT 계열 DNS 가 `*.trycloudflare.com` 을 못 푼다 | 와이파이로, 또는 폰 DNS 를 1.1.1.1 로 |
| MZOK 고정 주소 `/go/GB_jeton-nano-01/...` 가 `503 relay_unconfigured` | MZOK 공개 중계는 Worker 운영자가 기기별 `MZOK_RELAY_POLICIES` 를 넣어야 동작한다. 2026-11-03 기준 레지스트리의 모든 기기가 503 | 이 보드에서 고칠 수 없다. 레지스트리 조회(`/v1/devices/<id>`)는 토큰이 필요하므로 keeper 로그를 쓴다 |
| LAN `http://192.168.0.216/dave-face/` 에서 카메라가 안 켜진다 | 카메라는 HTTPS(보안 컨텍스트)에서만 열린다 | 터널 주소로. LAN http 는 curl 시험 전용 |
| 얼굴 위 메시가 안 보인다(인식은 됨) | WASM·모델 로드 실패 또는 GPU 델리게이트 실패 → 조용히 꺼진다 | 브라우저 콘솔 경고 1줄 확인, `curl -sI .../dave-face/assets/mediapipe/wasm/vision_wasm_internal.wasm` 이 `application/wasm` 200 인지 |
| 유사도가 계속 낮다(같은 사람) | 이 보드 DB 는 비어 있거나 Pi 에서 등록한 얼굴이 아니다 / 문턱 0.5 는 미보정 | `/dave-face/admin` 에서 재등록, `/compare` 로 분포 확인 |

현재 터널 주소(문서·보드에 적지 않는다):

```bash
grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' ~/works/baro_tunnel/localfiles/keeper-out.log | tail -1
```

## 주의 사항

- **데모다.** 라이브니스가 없어 사진·화면으로도 일치한다. 문턱 0.5 는 보정 전 값이다.
- `/api/recognize` 는 인증 없이 등록된 ID 와 점수를 돌려준다. 터널로 인터넷에 열려 있으니 얼굴은 동의한 사람만 등록하고, 관리자 카드·토큰은 아무에게나 주지 않는다.
- Face Mesh 는 `@mediapipe/tasks-vision` **0.10.35 고정**. 1.x 는 사용 통계를 Google(`odml.pa.googleapis.com`)로 보내고 옵트아웃이 없다 — 올리지 말 것. WASM·모델은 자체 호스팅(첫 로드 ≈11 MB, 폰에 캐시).
