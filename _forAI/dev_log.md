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
