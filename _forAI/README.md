# _forAI Guide

## 목차

- [한 줄 요약](#한-줄-요약)
- [읽는 순서](#읽는-순서)
- [문서 역할](#문서-역할)
- [현재 스냅샷](#현재-스냅샷)
- [유지 규칙](#유지-규칙)

## 한 줄 요약

이 디렉터리는 `dave_face`(얼굴 인식 성능 데모 — `jetson` 브랜치: Jetson Orin Nano 4GB + TensorRT, `main` 브랜치: Raspberry Pi 5 + Hailo-10H) 작업을 이어받을 때 필요한 AI 작업 문맥을 정리해 두는 곳이다.

## 읽는 순서

1. `README.md`
2. `inventory.md`
3. `memo.md`
4. `dev_log.md`
5. `plan.md`

## 문서 역할

- `inventory.md`: 저장소에 실제로 있는 구조, 엔트리포인트, 빌드/검증 명령을 기록한다.
- `plan.md`: 앞으로 진행할 개발 계획과 우선순위만 기록한다.
- `memo.md`: 프로토콜, 핀맵, 기본값, 디버깅 교훈 같은 참고 메모를 모은다.
- `dev_log.md`: 날짜별 작업 이력과 `_forAI` 정리 내역을 남긴다.

## 현재 스냅샷

- **2026-11-03 인수인계 확인:** 설치·포팅 검증은 완료되어 있고 현재 Dave Face는 대기 중이다. `oni-detector`가 실행 중이며 `dave-face`는 PM2 현재 목록·저장 목록에 없다. 따라서 3011 미리스닝과 `/dave-face/`의 HTTP 502는 이 운영 상태에서 예상되는 결과다.
- 흥의 시작 문서: [나노 보드 AI 서비스 인수인계](../../system_info/handover-nano-ai-20261103.md). 실행 전환, 현재 접속 주소 조회, 모델·데이터·토큰 위치를 함께 확인한다.
- 저장소 경로: `/home/vcacore/works/dave_face` (브랜치 `jetson`; GitHub `gbox3d/dave_face`, `main` = Pi 판 `/home/goback/works/dave_face`)
- 원격: `jetson` 브랜치는 origin(`gbox3d/dave_face`)에 올라가 있다(2026-11-03). 이 보드에서 push 가능(`gh` 기기 인증, `memo.md` 참고). 보드 허브는 바로메모 team-n #1797(oni-detector 와 함께).
- 설치 소스: `jetson` 브랜치의 2026-11-03 조회 HEAD `26bfcdb` / 포팅 기준 `d4dbdb1`, 애플리케이션 `0.1.0`. 2026-10-29 성능·GPU 검증 기록을 유지하며, 11월 3일에는 서비스를 전환하거나 GPU 벤치를 다시 실행하지 않았다.
- 대상 보드/플랫폼: NVIDIA Jetson Orin Nano **4GB**(p3767-0004, WorkGate 호스트 `ABP301`, 192.168.0.216), JetPack 6.0 / L4T 36.3.0, CUDA 12.2, cuDNN 8.9.4, TensorRT 8.6.2, Python 3.10.12, 전원 10W(이 모듈의 최대 모드)
- 현재 버전: 0.1.0 (루트 `package.json` · `apps/web/package.json` · `apps/api/pyproject.toml` · `dave_face.__version__` 네 곳이 같아야 한다)
- 메인 엔트리포인트: `apps/api/run-web.sh` → gunicorn `dave_face.web:create_app()` (127.0.0.1:3011, PM2 `dave-face`, nginx `/dave-face/`)
- 가속기 백엔드: `apps/api/dave_face/npu.py` = TensorRT 8.6 Python API + cuda-python 12.2, 엔진 `models/trt/*.fp16.trt8.6.2.engine`(이 보드에서 빌드, git 제외)
- 화면: `/` 인식 테스트 · `/compare` 사진 비교(데모 두 개, MediaPipe Face Mesh 오버레이) · `/admin` 등록(관리자). 사람용 사용법은 루트 `readme.md`
- GPU 공유: `oni-detector`(pm2)와 동시 실행 불가 — 전환 규칙은 `readme.md`·`memo.md`

## 유지 규칙

- 계획이 아닌 참고 정보는 `plan.md`가 아니라 `memo.md`에 둔다.
- 저장소 구조나 실행 명령이 바뀌면 `inventory.md`를 먼저 갱신한다.
- 작업 이력은 날짜를 붙여 `dev_log.md`에만 남긴다.
- 새 작업을 시작할 때는 `inventory.md`와 `memo.md`를 먼저 읽고, 실제 할 일은 `plan.md`에서 확인한다.
- 모든 문서에는 제목 바로 아래에 `## 목차` 섹션을 둔다.
- 사용자 동의 없이 git commit을 하지 않는다.
- 사용자 동의 없이 `_forAI/` 문서를 수정하지 않는다.
