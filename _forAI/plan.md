# Plan

## 목차

- [Current goal](#current-goal)
- [Near-term work](#near-term-work)
- [Structure decisions](#structure-decisions)
- [Risks](#risks)

## Current goal

- 라즈베리 + 헤일로 얼굴 인식 성능을 폰으로 보여 주는 데모를 안정적으로 유지한다 (인식 테스트 · 사진 비교 · 등록).

## Near-term work

- 폰(iPhone Safari)에서 사진 비교 화면 실사용 확인: 같은 사람·다른 사람·거리·각도별 유사도 분포를 보고 `MATCH_THRESHOLD`(0.5)를 정한다.
- 터널 주소가 바뀌면 `scripts/print-qr.py` 로 접속 QR 카드를 다시 뽑는다.

## Structure decisions

- NPU 의존은 `apps/api/dave_face/npu.py` 하나로 가둔다. 다른 보드(RK3588·QCS6490·CIX P1 등)로 옮길 때는 모델 변환 + `npu.py` 교체만 한다.
- 웹은 Vite 다중 엔트리(화면별 HTML)와 상대 경로만 쓴다. 공용 로직은 `flow.tsx`.
- 서버 상태는 등록 임베딩(sqlite)뿐이다. 사진 비교는 서버에 저장하지 않는다.
- 보류 (요청이 있을 때만): 라이브니스·패시브 PAD, 1:1 인증, CCTV RTSP 입력, 대규모 임계값 보정.

## Risks

- 퀵터널 주소가 재시작마다 바뀐다 → 인쇄한 QR 이 무효가 된다. KT 계열 DNS 는 `*.trycloudflare.com` 을 막을 수 있다.
- `/api/recognize`·`/api/embed` 가 무인증으로 인터넷에 열려 있다 → 등록 ID·점수가 보이고 NPU 를 누구나 쓸 수 있다 (데모라 감수).
- NPU 프로세스 독점 → dave-face 와 piWanKanobi 를 동시에 돌릴 수 없다 (`pm2 stop`/`start` 로 전환).
- 문턱 0.5 는 미보정이고, 라이브니스가 없어 사진·화면으로도 일치한다.
