# Plan

## 목차

- [Current goal](#current-goal)
- [Near-term work](#near-term-work)
- [Structure decisions](#structure-decisions)
- [Risks](#risks)

## Current goal

- 2026-11-03: 설치·포팅 검증을 마친 Dave Face를 흥이 Oni와 교대로 사용할 수 있도록 인계한다. 현재 Dave는 대기 중이고 Oni가 실행 중이다. 첫 실행은 [나노 보드 AI 서비스 인수인계](../../system_info/handover-nano-ai-20261103.md)의 전환 순서를 따른다.
- Jetson Orin Nano 4GB(ABP301)에서 Pi 와 같은 모델로 얼굴 인식 데모를 안정적으로 보여 준다(인식 테스트 · 사진 비교 · 등록 + Face Mesh 연출). `main`(Pi/Hailo)과 수치를 나란히 비교한다.

## Near-term work

- 인수자가 사용할 때: 현재 HTTPS 주소를 조회하고 Oni를 정지한 뒤 Dave를 기동하여 health `ready`를 확인한다. 동의한 얼굴을 등록하고 데모를 사용한 후 Dave를 PM2 목록에서 제거하고 Oni를 복원한다. 이 문서 정리에서는 전환 시험을 수행하지 않았다.
- 폰(iPhone Safari) 실사용: 메시 오버레이가 카메라 위에 맞게 그려지는지(미러·cover 정렬), 첫 로드 시간(WASM ≈11 MB), GPU 델리게이트 실패 시 CPU 폴백 동작 확인.
- `MATCH_THRESHOLD`(0.5) 를 FP16 엔진 기준으로 재검증 — 같은 사람·다른 사람 분포를 `/compare` 로 보고 정한다. 이 보드 DB 는 비어 있으니 `/admin` 재등록.
- 측정 보강: 콜드 스타트(30 s 이상 유휴 뒤 첫 요청 — GPU 클럭 306 MHz 유휴·스왑) 수치를 readme 에 추가.
- 터널 주소가 바뀌면 `scripts/print-qr.py` 로 접속 QR 카드를 다시 뽑는다.
- 선택: INT8(캘리브레이션) 또는 `scrfd_2.5g` 엔진(디코더 동일) 으로 지연 비교; `arcface_r50` 인식기(LFW 99.7) — Pi 도 같은 Zoo 모델이 있어 양쪽에서 바꿀 때만.
- 선택: Pi 와 Jetson 에서 같은 JPEG 의 `/api/embed` 임베딩 코사인(≥0.95 기대) 실측해 기록.

## Structure decisions

- 가속기 의존은 `apps/api/dave_face/npu.py` 하나로 가둔다(Hailo → TensorRT 교체가 그 파일 + 모델 스크립트로 끝났다). 다른 보드로 옮길 때도 같다.
- 모델은 Hailo Model Zoo 원본 ONNX 를 쓴다 — Pi 와 가중치가 같아 비교가 성립한다. 엔진은 보드에서 빌드하고 커밋하지 않는다(TRT 버전 포함 파일명).
- 웹은 Vite 다중 엔트리(화면별 HTML)와 상대 경로만 쓴다. 메시는 데모 두 페이지에만, `mesh.ts` 한 파일 + `Stage` 프롭. 와이어 키 `npu_ms` 는 유지(라벨만 GPU).
- 서버 상태는 등록 임베딩(sqlite)뿐이다. 사진 비교·메시는 서버에 아무것도 남기지 않는다.
- 운영: GPU 를 `oni-detector` 와 배타적으로 쓴다(인터락 + 전환 규칙). dave-face 는 pm2 덤프에 넣지 않는다.
- 보류(요청이 있을 때만): 라이브니스·패시브 PAD, 1:1 인증, CCTV RTSP 입력, 대규모 임계값 보정, web.py 의 초기화 실패 fail-fast(`os._exit(3)`), 액세스 로그 회전.

## Risks

- 4 GB 메모리: 두 GPU 앱 동시 기동(OOM) — 인터락·덤프 규칙으로 막지만 사람이 `pm2 save` 를 잘못 하면 재발. 워커 OOM 은 gunicorn 마스터가 재포크(그동안 health 503).
- 초기화 실패(엔진 없음·GPU 점유)는 `state: error` 503 으로 굳는다(web.py 불변) → `pm2 restart dave-face` 로 복구, run-web.sh 사전 검사로 대부분 사전 차단.
- 퀵터널 주소가 재시작마다 바뀐다 → 인쇄한 QR 이 무효가 된다. KT 계열 DNS 는 `*.trycloudflare.com` 을 막을 수 있다.
- `/api/recognize`·`/api/embed` 무인증(데모) — 등록 ID·점수가 보이고 GPU 를 누구나 쓸 수 있다.
- 문턱 0.5 미보정, 라이브니스 없음.
- JetPack 업그레이드 시 엔진 재빌드 필요(파일명에 TRT 버전 → "엔진 없음" 으로 드러남), cuda-python 핀(12.2.*) 도 드라이버에 맞춰 바꿔야 한다.
