# Memo

## 목차

- [인수인계 현황 (2026-11-03)](#인수인계-현황-2026-11-03)
- [제품 기준선](#제품-기준선)
- [기본 설정값](#기본-설정값)
- [데이터 계약 (모델·엔진)](#데이터-계약-모델엔진)
- [런타임 구조 메모](#런타임-구조-메모)
- [호스트·운영 메모 (ABP301)](#호스트운영-메모-abp301)
- [대안 조사 (2026-10-28)](#대안-조사-2026-10-28)
- [동작 규칙](#동작-규칙)
- [반복 금지](#반복-금지)

## 인수인계 현황 (2026-11-03)

- 흥의 통합 운영 문서: [나노 보드 AI 서비스 인수인계](../../system_info/handover-nano-ai-20261103.md).
- Dave Face `0.1.0` / `jetson`은 설치 완료·대기 상태다. 2026-11-03 조회 HEAD는 `26bfcdb`, 포팅 기준은 `d4dbdb1`이다. 기본 실행 서비스는 `oni-detector`; Dave는 PM2 현재 목록과 재부팅용 저장 목록에 없다. 3011 연결 실패와 `/dave-face/` 502는 대기 상태의 결과이며 재설치 근거가 아니다.
- ONNX/ZIP 해시 4개, FP16 엔진 2개, 웹 빌드, 가상환경을 확인했다. 등록 임베딩은 0개이므로 사용할 얼굴은 동의를 받고 `/admin`에서 등록한다. 관리자 토큰 값과 얼굴 데이터는 문서·보드에 복사하지 않는다.
- 사용할 때는 먼저 `pm2 stop oni-detector` 후 이 저장소의 `deploy/ecosystem.config.cjs`로 Dave를 시작한다. 끝나면 `pm2 delete dave-face` 후 `pm2 start oni-detector`로 복원한다. `pm2 delete oni-detector`는 하지 않으며, Dave가 목록에 있는 동안 `pm2 save`를 실행하지 않는다. 전체 명령은 루트 `readme.md`와 통합 운영 문서에 있다.
- 접속 호스트는 고정하지 않는다. MZOK 장치 `GB_jeton-nano-01`의 현재 주소 조회 절차는 통합 운영 문서를 따른다. 로컬 확인 위치는 `~/works/baro_tunnel/localfiles/keeper-out.log`다. 상대 경로 `/dave-face/`, `/dave-face/compare`, `/dave-face/admin`을 현재 HTTPS 주소에 붙인다.
- 아래 성능 및 44개 테스트 통과 수치는 **2026-10-29 기록**이다. 11월 3일에는 문서 정리와 읽기 전용 상태 확인만 수행했고, 서비스 전환·빌드·GPU 벤치마크는 실행하지 않았다.

## 제품 기준선

- 보드: NVIDIA Jetson Orin Nano **4GB** 모듈(p3767-0004) + Orin NX 레퍼런스 캐리어 = WorkGate 호스트 `ABP301`(192.168.0.216). JetPack 6.0 / L4T 36.3.0, CUDA 12.2.140, cuDNN 8.9.4, TensorRT 8.6.2.3, Python 3.10.12, Ubuntu 22.04. 전원 모드 10W(이 모듈·JetPack 6.0 의 최대; `nvpmodel`·`jetson_clocks` 변경 금지).
- 가속기 백엔드: `npu.py` = TensorRT 8.6 Python API(`tensorrt` 시스템 패키지) + `cuda-python==12.2.*`(cudart, `cython` 필요). 엔진 FP16, I/O fp32, 배치 1. 엔진 파일 `models/trt/<name>.fp16.trt8.6.2.engine`(TRT 버전·SM 8.7 전용, git 제외, `scripts/build-engines.sh`).
- 모델(Hailo Model Zoo 원본 float ONNX — Pi 의 HEF 와 같은 가중치, InsightFace 출신, ms1mv3, LFW 99.43):
  - `scrfd_10g.zip`(15,714,580 B) → `scrfd_10g_bnkps.onnx`(16,923,479 B) sha256 `ba018cc303bdb3279c102794db84c0b5e86ebf32a5d765c63a11138f52917aca`
  - `arcface_mobilefacenet.zip`(7,696,419 B) → `mbf.onnx`(8,179,605 B) sha256 `1ef27ee20b9264f00f65d59b2be340001a012ffd686fe6b835ec0d702820f5c9`
  - 전체 해시는 `models/onnx/SHA256SUMS`. 출처 S3 `hailo-model-zoo.s3.eu-west-2.amazonaws.com/{FaceDetection/scrfd/scrfd_10g,FaceRecognition/arcface/arcface_mobilefacenet}/pretrained/...`.
- Python: 시스템 cv2 4.8.0(CUDA 없음) · numpy 1.21.5 · Pillow 9.0.1 · scipy 1.8 + venv Flask 3.1 · gunicorn 23 · segno · tomli. Node 24.22 + pnpm 11.1, React 19 · Vite 8 · TS 7 · `@mediapipe/tasks-vision` **0.10.35**(고정) · qrcode-generator 2.0.4.
- 실측 (2026-10-29, 10W, oni-detector 정지 상태): SCRFD GPU compute 14.8 ms / Python `infer` 30.7 ms, ArcFace 2.0 / 3.5 ms, `/api/recognize` 서버 58 ms(`npu_ms` 41.8), 워커 RSS 462 MB, 엔진 빌드 mbf 78 s · scrfd 192 s. 비교: Pi 5 + Hailo-10H NPU 41 + 6.7 ms, 서버 67 ms, 워커 ≈200 MB.

## 기본 설정값

| 값 | 위치 | 설명 |
|---|---|---|
| 포트 3011 | `run-web.sh`, `deploy/*` | 127.0.0.1 바인드. nginx `/dave-face/` → `127.0.0.1:3011/` |
| `MATCH_THRESHOLD = 0.5` | `web.py` | 코사인이 이 이상이면 "일치". 보정 전 값(FP16 에서 재검증 필요) |
| JPEG ≤ 1 MiB, 긴 변 240–1920, 짧은 변 ≥ 120 | `web.py` | 아니면 413/422 |
| 캡처 긴 변 1280, 품질 0.85 | `capture.ts` | 업로드 픽셀은 비미러, 미리보기만 CSS 미러 |
| SCRFD 점수 0.5, NMS 0.4 | `scrfd.py` | 가장 큰 얼굴 하나만 쓴다 |
| 엔진 타임아웃 5 s, 슬롯 대기 1 s | `web.py` `InferenceEngine` | 붐비면 503 busy → 클라이언트가 200 ms 뒤 재시도 |
| 관리자 토큰 | `localfiles/admin-token` (0600, 첫 실행 때 생성) | 접속 QR 은 `/admin#t=<토큰>` |
| 터널 주소 | `~/works/baro_tunnel/localfiles/keeper-out.log` 의 마지막 trycloudflare 호스트 | 재시작마다 바뀐다(MZOK id `GB_jeton-nano-01`). 문서·보드에 적지 않는다 |
| 메시 자산 | `apps/web/public/assets/mediapipe/{wasm/,face_landmarker.task}` | `scripts/fetch-mediapipe.sh`(SHA-256 고정). 브라우저는 `./assets/mediapipe/…` 로 받는다 |
| 엔진 빌드 플래그 | `scripts/build-engines.sh` | `--fp16 --memPoolSize=workspace:256 --maxAuxStreams=0 --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT --timingCacheFile --skipInference --verbose`, shape 플래그 없음 |
| 메모리 게이트 | `build-engines.sh` `MIN_AVAILABLE_KB=1500000` | 빌더 라이브러리 고정비 때문 |

## 데이터 계약 (모델·엔진)

- SCRFD(`scrfd_10g_bnkps.onnx`): **정적 입력 `input.1` 1×3×640×640 float32**(InsightFace 배포본과 다른 파일 — PyTorch 1.10 export, onnxsim). 출력 9개 `score_8/16/32`(N×1), `bbox_8/16/32`(N×4), `kps_8/16/32`(N×10), shape `(1, N, C)`, N = 12800/3200/800, 앵커 인덱스 최하위. **점수는 그래프 안 Sigmoid 를 지난 값**(Hailo 는 conv 에서 끊고 alls 로 시그모이드를 다시 붙였다), bbox/kps 는 스트라이드 단위(`scrfd.decode` 가 곱한다). 호스트 정규화 `(x−127.5)/128`, RGB.
- ArcFace(`mbf.onnx`): 입력 `data` `(-1, 3, 112, 112)`(동적 배치 — trtexec 가 1 로 고정), 출력 `452` `(1, 512)` 비정규화. 호스트 정규화 `(x−127.5)/127.5`, RGB. L2 는 `embed()`.
- `npu.py` 가 `(1,N,C)` → `rows.reshape(n, n, 2C)` 로 바꾸면 HEF 의 (H,W,C) 레이아웃과 같아 `scrfd._heads`(shape `(n, 2|8|20)` 매칭)는 변경 없음. 로드 때 단언: 출력 9개 n∈{80,40,20}, 점수 헤드 ∈[0,1], 인식기 출력 1개 512.
- trtexec 8.6: 정적 모델에 `--minShapes/--optShapes/--maxShapes` 를 주면 **에러**("Static model does not take explicit shapes"). 동적 축은 shape 없이 빌드하면 1 로 자동 대체(경고). TRT 8.6 은 cuDNN/cuBLAS 전술을 기본으로 끈다(`--tacticSources` 는 명시용).
- `deserialize_cuda_engine` 은 실패해도 None 만 돌려준다(이유는 ILogger). `cudart` 호출은 `(err, *values)` 튜플, `cudaGetErrorString` 은 `(err, bytes)`. 스트림 핸들은 `int(stream)`(`getPtr()` 은 포인터의 포인터). CUDA 컨텍스트는 `trt.Runtime` 생성 때, fork 뒤 자식에서는 못 쓴다(`cudaErrorInitializationError`).

## 런타임 구조 메모

- 요청 흐름: 폰(HTTPS 터널) → nginx `/dave-face/` → gunicorn 워커(스레드 4) → `InferenceEngine` 큐(크기 1) → 추론 스레드가 `FacePipeline.analyze` → 응답(`elapsed_ms`, `npu_ms`). `FaceNPU` 는 추론 스레드에서 만들고 닫는다.
- `npu_ms` 는 Pi 와 달리 호스트 정규화·NCHW 변환(numpy, 640² 에서 5~10 ms)·H2D/D2H·동기화를 포함한다. GPU 만의 시간은 `trtexec --loadEngine` 의 GPU Compute Time.
- 워밍업: `FaceNPU.__init__` 이 두 엔진을 한 번 돈다(CUDA 지연 로딩·첫 커널 적재) → health 가 `ready` 되기까지 수 초~수십 초.
- 메시: `flow.tsx` `Stage` 가 `mesh` 일 때 `<canvas class="mesh">` 를 비디오 위에 두고 `useFaceMesh`(mesh.ts)가 rAF 루프에서 `detectForVideo` → `DrawingUtils`. 캔버스는 비디오와 같은 CSS(`object-fit: cover; transform: scaleX(-1)`)라 좌표 변환이 없다. 로딩·추론 실패는 조용히 비활성(인식 루프 무관). 패키지는 동적 import 로 별도 청크(관리자 페이지는 안 받는다).
- PM2: 앱 `dave-face`(fork, 1 instance, `exp_backoff_restart_delay`·`max_restarts 10`·`min_uptime 30 s` — 사전 검사 exit 2 가 무한 루프 대신 errored 로). `pm2 start` 는 호출 셸 env 를 복사하므로 `env -i …` 로, pm2 는 절대 경로로 띄운다.
- WorkGate 카드 "Dave Face"(열기 `/dave-face/`, 링크 사진 비교·등록). 갱신 `node ~/works/WorkGate/scripts/py.cjs registry put ./deploy/workgate.json --by …`. dave-face 가 내려가 있으면 행이 빨갛게 보이는 것이 정상.

## 호스트·운영 메모 (ABP301)

- **GPU 공유**: `oni-detector`(pm2, oni_detector 1.5.0, torch 2.3.0 NVIDIA 휠 + onnxruntime, `~/works/_ai_works/oni_detector-1.5.0-5c5c11d`, `.venv-jetson`, 설치 문서 `docs/INSTALL-JETSON.md`)가 혼자 돌 때 nvmap GPU 메모리 1.42 GB(스왑 불가) + VmSwap 1.2 GB, 보드 MemAvailable ≈290 MB(2026-11-03 실측). oni 는 `.env` `ONI_HOST=0.0.0.0` 이라 `0.0.0.0:9082` 로 듣는다(oni 소관, 바꾸지 않음). dave-face(워커 RSS 0.4~0.8 GB + GPU)와 동시 실행 불가. `run-web.sh` 가 `:9082` 가 열려 있으면 기동을 거부한다. 전환: `pm2 stop oni-detector` ↔ `pm2 delete dave-face && pm2 start oni-detector`. `pm2 delete oni-detector` 금지(stop 만).
- **pm2 데몬은 `pm2-vcacore.service`(system.slice)에서 돈다**(2026-10-29 이동, drop-in `OOMPolicy=continue`). 그 전에는 VS Code 터널 cgroup(`code-tunnel.service`, OOMPolicy=stop)에 있어 OOM 1건이 pm2 전체·터널을 멈출 수 있었다. 에이전트 셸은 여전히 그 cgroup 이므로 무거운 작업은 `systemd-run --user --unit=… --collect --same-dir -p OOMPolicy=continue -p MemoryMax=… -- nice -n 10 choom -n 1000 <cmd>` 로 격리한다(유닛 PATH 에 `~/.local/bin` 없음 → 스크립트가 `uv` 절대 경로를 찾는다).
- **`pm2 save` 는 dave-face 가 목록에 없을 때만.** PM2 7.1.1 `resurrect` 는 저장된 앱을 상태와 무관하게 띄우므로 두 GPU 앱이 부팅 때 같이 뜰 수 있다.
- 엔진 빌드: trtexec 빌더 라이브러리 고정비 ≈1–2 GB → 가용 메모리 1.5 GB 게이트, mbf → scrfd 순차, 로그 `localfiles/logs/trt-build-*.log`. OOM 진단은 `journalctl -k -b | grep -iE 'oom-kill|killed process'`(sudo 불필요, adm 그룹).
- nginx: `sites-available/default` 에 include 한 줄, 변경은 백업 → `sudo nginx -t` → `sudo systemctl reload nginx`(restart 금지, 실패하면 전 서비스가 내려간다). 저장소 이동/삭제 전에 include 제거.
- 폰 접속: HTTPS 터널만 카메라 가능. 주소는 keeper 로그(MZOK `GB_jeton-nano-01`), KT 계열 DNS 는 `*.trycloudflare.com` NXDOMAIN 가능. LAN http 는 curl 테스트 전용.
- 터널 주소는 `baro-tunnel-auto` 가 재시작할 때마다 바뀐다 — 2026-10-29 pm2 데몬 이동 때 한 번 바뀌어 "WorkGate 터널 주소가 안 들어간다" 문의가 있었다(터널·`/gate/` 는 정상, 옛 주소였다). MZOK 레지스트리는 keeper 가 자동 갱신한다. MZOK 공개 중계 `GET /go/<id>/...` 는 **503 `relay_unconfigured`**(2026-11-03, 레지스트리의 모든 기기 공통 — Worker 운영자가 `MZOK_RELAY_POLICIES` 를 넣어야 동작), 레지스트리 조회 `/v1/devices/<id>` 는 토큰 필요 → 로컬에서는 keeper 로그를 쓴다.
- Git: 2026-11-03 부터 이 보드에서 push 가능 — `gh` 2.103.3 을 `~/.local/bin/gh` 에 설치(공식 릴리스, 체크섬 확인)하고 gbox3d 계정 기기 인증(`gh auth login --web`, scope `repo`), `gh auth setup-git` 으로 git credential helper 를 연결했다. 토큰은 `~/.config/gh/hosts.yml` 평문(키링 없음) — 회수는 `gh auth logout`. 그 전에는 자격증명이 없어 `could not read Username` 으로 실패했다.
- 바로메모(team-n) #1780 이 이번 포팅의 작업 글(시작/결과). **#1797 이 이 보드의 허브**(oni-detector 와 dave-face 공유 규칙·실측·미결 항목, 댓글로 갱신).

## 대안 조사 (2026-10-28)

| 대안 | 결론 |
|---|---|
| **A. TensorRT 네이티브**(채택) | 설치된 런타임만 사용, 메모리 최소. cuda-python 12.2.1 aarch64 휠 PyPI 존재 |
| B. ONNX Runtime GPU | JetPack 6.0(CUDA 12.2)용 휠은 Jetson Zoo(1.17–1.19, 봇 차단 위키)만, jp6/cu126 인덱스는 6.1+ 전용. cuDNN 전체 적재. 최후 수단 |
| C. DeepStream 7.0 / D. jetson-inference·Docker / G. torch2trt | 용도 불일치·메모리. Polygraphy(순수 Python, 자체 ctypes cudart)는 예비 #2, ctypes-libcudart 직접 호출이 예비 #1 |
| E. CPU onnxruntime | 0.3–0.7 s/프레임 — 서비스 불가, 엔진 대조용만 |
| F. 다른 모델(YuNet·YOLO-face·AdaFace…) | Pi 비교 불가 → 제외. 같은 Zoo 의 `arcface_r50`(LFW 99.7)·`scrfd_2.5g`(디코더 동일, ≈3.9× 적은 연산)만 옵션 |
| 메시 `@mediapipe/tasks-vision` | 1.x 번들에 `odml.pa.googleapis.com/v1/log` 텔레메트리(옵트아웃 없음, 이슈 #6306) → 0.10.35 고정, 자체 호스팅 |

## 동작 규칙

- 데모다: 인증·라이브니스·시도 제한이 없다. `/api/recognize`·`/api/embed` 는 무인증(등록된 ID 와 점수를 돌려준다). 등록·목록·삭제만 관리자 토큰(`X-Admin-Token`, 빈 토큰이면 거부).
- 얼굴이 하나라도 보이면 항상 임베딩한다(가장 큰 얼굴). 크기·선명도·yaw/roll 은 값만 보여 준다.
- 등록은 사진 한 장, 같은 ID 는 교체. DB 스키마를 바꾸면 마이그레이션 없이 파일을 지우고 재시작한다.
- 사진·크롭은 저장하지 않는다. 임베딩만 `localfiles/dave_face.db`(0600). 얼굴은 동의한 사람만 등록한다. 메시 입력은 폰 안에서만 처리된다(외부 요청 없음).

## 반복 금지

- `@mediapipe/tasks-vision` 을 1.x 로 올리지 말 것 — 사용 통계를 Google 로 보내고 옵트아웃이 없다(0.10.35 번들에는 없음, 직접 grep 확인).
- `pm2 save` 를 dave-face 가 목록에 있을 때 하지 말 것; `pm2 delete oni-detector` 금지(stop 만); bench·trtexec `--loadEngine` 은 dave-face 를 내린 뒤에만(엔진·CUDA 컨텍스트 두 벌 = 4 GB 초과).
- trtexec 에 정적 모델 shape 플래그를 주지 말 것(에러). `cuda-python` 을 `--no-deps` 로 깔지 말 것(`import cython` 실패). `trt.OnnxParser` 로 ONNX 를 "가볍게" 들여다보려 하지 말 것 — Builder 가 빌드급 메모리를 쓴다(trtexec `--verbose` 로그를 읽는다).
- 에이전트 셸(VS Code 터널 cgroup)에서 엔진 빌드·pnpm install/build 를 직접 돌리지 말 것 — `systemd-run --user` 격리 유닛으로.
- gunicorn `--preload` 금지(CUDA 컨텍스트는 fork 를 못 건넘). `web.py` 에서 모듈 import 때 tensorrt 를 건드리지 말 것(추론 스레드 안에서만).
- 데모 요청에 요청하지 않은 보안 층을 넣지 말 것(Pi 에서 챌린지·잠금을 넣었다가 유사도가 한 번도 안 나왔다, 2026-10-27). 실기기에서 검증 안 된 문턱을 핵심 경로 앞에 두지 말 것.
- Vite 엔트리를 컴포넌트와 대소문자만 다른 이름으로 만들지 말 것(`admin.tsx` vs `Admin.tsx` → TS1149). 페이지 경로 끝에 `/` 를 붙이지 말 것(`/admin/` 은 404 로 둔다).
- 백그라운드 점검 스크립트는 `sys.stdout.reconfigure(line_buffering=True)`. nginx reload 직후 첫 요청은 옛 워커로 갈 수 있다 — 한 번 더 확인. iOS Safari 는 화면 전환 때 fetch 를 끊는다("Load failed") — `pageshow` persisted 때 오류를 지운다. fps 는 마지막 프레임 시각으로.
- 터널 호스트·관리자 토큰·`#t=` URL 을 문서·보드·채팅에 적지 말 것.
- 폰에서 주소가 안 열린다고 터널(`baro-tunnel-auto`)이나 pm2 데몬을 재시작하지 말 것 — 주소가 또 바뀐다. 먼저 keeper 로그의 최신 주소로 `/gate/` 가 200 인지 본다.
