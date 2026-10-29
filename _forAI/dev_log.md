# Dev Log

## 목차

- [Entries](#entries)

## Entries

### 2026-10-26

- 제안서(Hailo-8 기준 FaceNet → SCRFD + ArcFace) 검토. Hailo-10H 는 HailoRT 5.4.0 InferModel API 만 동작함을 확인하고, Model Zoo v5.4.0 hailo10h HEF 로 계획 (`~/.claude/plans/snug-baking-parnas.md`).

### 2026-10-27

- 모노레포 골격 + `_forAI/` 템플릿, HEF 고정(`SHA256SUMS`), uv venv, NPU 첫 접촉(scrfd_10g 41 ms, arcface 6.7 ms, 프로세스 독점 확인). SCRFD 디코드·정렬·파이프라인·DB·웹·React 페이지 구현.
- 배포: PM2 `dave-face`, nginx `/dave-face/` include, WorkGate 등록. piwankanobi 를 `pm2 delete` 했다가 사용자 정정으로 정지 상태로 복원.
- 1차 수정 요청: ID 없는 로그인(1:N), 관리자/데모 화면 분리, 응답 시간 표시, 접속 QR(관리자 토큰 포함) + LAN 프린터 인쇄(`scripts/print-qr.py`).
- 폰 실사용에서 고개 돌리기 챌린지가 4번 모두 타임아웃 → 사용자 요청으로 인증 기능 전부 제거(챌린지·세션·잠금·1:N 로그인·8장 등록·품질 거름). 사진 한 장 등록 + `/api/recognize`(프레임마다 유사도·시간)로 단순화. 등록돼 있던 dave1 은 옛 갤러리로 옮겼다.
- 사진 비교 화면(`/compare`, `/api/embed`) 추가 — 기준 사진 한 장과 카메라 얼굴의 유사도를 즉석에서 본다. 화면 셋을 잇는 메뉴.
- 교차 점검(검토 3명) 반영: 375px iPhone 에서 점수가 화면 밖으로 밀림(카메라 높이 45svh), 정지 후 fps 변동, 탭하기 전에 찍힌 프레임이 기준이 되던 문제, Safari "Load failed" 잔상.
- `_forAI/` 문서 채움 (사용자 요청 "문서 정리 하시고 커밋하시죠"), 첫 커밋.

### 2026-10-28 ~ 29 — `jetson` 브랜치: Jetson Orin Nano 4GB(ABP301) 포팅

- 조사: Hailo Model Zoo 가 HEF 의 원본 float ONNX(`scrfd_10g_bnkps.onnx`, `mbf.onnx`)를 공개 → Pi 와 같은 가중치로 TensorRT FP16 엔진. 대안(ONNX Runtime GPU·DeepStream·CPU ORT·다른 모델) 비교 후 TensorRT 네이티브 채택(`memo.md`). 7-에이전트 플랜 검토로 초안 오류 교정(trtexec shape 플래그, cuda-python `--no-deps`, `pm2 save`, Builder 기반 ONNX 조사).
- 보드 운영: pm2 데몬을 VS Code 터널 cgroup(OOMPolicy=stop)에서 `pm2-vcacore.service`(drop-in `OOMPolicy=continue`)로 이동(사용자 승인). 무거운 단계는 `systemd-run --user` 격리 유닛으로. `oni-detector` 는 빌드·테스트 동안 stop 후 복원.
- 코드: `npu.py` 를 TensorRT 8.6 + cuda-python 12.2 백엔드로 교체(계약 유지, 로드 단언·워밍업), `scripts/{setup-venv,fetch-onnx,build-engines}.sh` 신규, `run-web.sh` 사전 검사(엔진·`:9082` 인터락·포트·`umask 077`·`oom_score_adj`), ecosystem 백오프, workgate.json 경로, Python 3.10 호환(tomli), bench 수정. `web.py`·`scrfd.py`·`pipeline.py`·`db.py` 본문 불변.
- 웹: 데모 두 페이지에 MediaPipe Face Mesh 오버레이(`mesh.ts`, `Stage` 캔버스, `@mediapipe/tasks-vision` 0.10.35 고정 — 1.x 는 텔레메트리), 라벨 NPU→GPU, 문구 정리. WASM·모델 자체 호스팅(`scripts/fetch-mediapipe.sh`).
- 결과: 엔진 빌드 mbf 78 s · scrfd 192 s; `pnpm test:api` 44건 OK; bench SCRFD 30.7 ms(GPU 14.8) · ArcFace 3.5 ms(GPU 2.0), lena 1명·t1 6명, 동일 사진 cos 1.000·타인 0.06; 서비스 ready 3 s, `/api/recognize` 중앙값 58 ms(`npu_ms` 41.8), 워커 RSS 462 MB; nginx `/dave-face/` 200, WorkGate 등록·health 200, 터널 경유 확인.
- 문서: `readme.md`·`_forAI/` 전부 Jetson 기준으로 갱신(사용자 요청 `/forai-scaffold`). 바로메모 team-n #1780.
- 마무리: 검증 후 `pm2 delete dave-face` → `pm2 start oni-detector` 로 원상복구(`pm2 save` 없음), 커밋 1회.
