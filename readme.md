# Dave Face Recognition System

Raspberry Pi 5 + Hailo-10H NPU **얼굴 인식 데모**. 사진 한 장으로 얼굴을 등록하고, 카메라 프레임마다 등록된 얼굴들과의
유사도와 처리 시간(NPU·서버·왕복)을 실시간으로 보여 준다. 라즈베리 + 헤일로 성능을 보는 용도라 인증·보안 기능은 없다.

- 파이프라인: SCRFD-10G 검출(5점) → 112×112 정렬 → ArcFace MobileFaceNet 512-D → 코사인 유사도 (Model Zoo v5.4.0 hailo10h HEF)
- `apps/api`: Python 3.13, Flask + gunicorn, HailoRT 5.4.0 InferModel API — NPU 는 워커 스레드 하나가 쥔다
- `apps/web`: React 19 + Vite 8 — `/` 인식 테스트, `/compare` 사진 비교, `/admin` 등록(관리자: 등록·삭제·접속 QR). 화면 위 메뉴로 오간다
- 운영: PM2 `dave-face` · 포트 3011 · nginx `/dave-face/` · WorkGate 카드 "Dave Face"

## 사용법 (폰)

폰 카메라는 **HTTPS 에서만** 열린다 → baro_tunnel 퀵터널 주소로 연다(터널이 재시작하면 주소가 바뀐다).
LAN `http://192.168.0.213/dave-face/` 로 열면 카메라가 안 된다.

**접속 QR 카드** — 테스트 카드와 관리자 카드(토큰이 들어 있어 자동 입력)를 뽑아 폰으로 찍는다:

```bash
cd ~/works/dave_face
apps/api/.venv/bin/python scripts/print-qr.py              # 터널 주소·토큰 → localfiles/qr/cards.pdf → LAN IPP 프린터
apps/api/.venv/bin/python scripts/print-qr.py --no-print   # PDF 만
```

수동으로: 터널 주소 `grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' ~/works/baro_tunnel/localfiles/keeper-out.log | tail -1`
뒤에 `/dave-face/`(테스트) 또는 `/dave-face/admin`(관리자 — 끝에 `/` 없이). 토큰은 `cat localfiles/admin-token`.

1. **등록 (관리자)** — ID 입력 → **카메라 켜서 등록** → 얼굴이 잡히면 **촬영해 등록**. 사진 한 장. 같은 ID 는 사진이 바뀐다.
   정면으로 가까이 찍을수록 나중 유사도가 높다.
2. **테스트** — **인식 시작** → 프레임마다 등록된 모두와의 유사도 막대(높은 순), 문턱 이상이면 "일치 · id",
   얼굴 값, 시간(fps · 서버 · NPU · 왕복). **정지**로 끝낸다.
3. **사진 비교** (`/dave-face/compare`, 등록·토큰 없음) — **카메라 켜기** → **기준 사진 찍기** → 그 뒤로 카메라에 비치는
   얼굴과 기준 사진의 유사도를 실시간으로 본다. 다른 사람·거리·각도로 바로 비교. **기준 다시 찍기**로 바꾼다.
   아무것도 저장하지 않는다 (임베딩만 받아 브라우저가 코사인을 잰다).

## 수치 보는 법

| 값 | 뜻 |
|---|---|
| 유사도 | ArcFace 임베딩의 코사인 (−1~1). 문턱 0.5 이상이면 "일치". 보정 전 초기값이라 이 화면으로 같은 사람·다른 사람 분포를 보고 정한다 (`apps/api/dave_face/web.py` `MATCH_THRESHOLD`) |
| NPU | SCRFD + ArcFace 두 번의 NPU 호출 시간 |
| 서버 | Pi 가 요청을 받아 답하기까지 (JPEG 디코드 + NPU + 비교) |
| 왕복 | 폰이 잰 요청~응답 (터널·네트워크 포함) |
| 눈 거리 · yaw · roll · 선명도 | 얼굴 크기(px) · 좌우 돌림 비율(정면 ≈ 0) · 기울기(°) · Laplacian 분산. 보여 주기만 하고 거르지 않는다 |

실측 (2026-10-27, nginx 경유 20회 중앙값): 서버 67 ms · NPU 53 ms (SCRFD 41 + ArcFace 6.7). 공개 샘플 사진
(`localfiles/samples/public`): 같은 사람 다른 해상도 0.980, 같은 사진 50%·35% 축소(눈 사이 36·25 px) 0.946·0.903,
다른 사람 −0.02~0.14.

## 저장소 구조

```
apps/api/dave_face/   npu.py       VDevice 1개 + InferModel 2개 (hailo_platform 을 import 하는 유일한 모듈)
                      scrfd.py     letterbox, 헤드 (H,C) 매칭, 앵커 디코드, NMS
                      pipeline.py  Umeyama 정렬, 선명도·자세 값, FacePipeline.analyze → Obs | Reject("no_face")
                      db.py        sqlite3 users(사용자당 임베딩 하나), 관리자 토큰
                      web.py       InferenceEngine + REST + 페이지 서빙 (/ 테스트, /compare 사진 비교, /admin 등록)
apps/api/tests/       unittest — NPU 를 열지 않는다 (가짜 엔진 주입)
apps/web/src/         Recognize.tsx(테스트) · Compare.tsx(사진 비교) · Admin.tsx · Qr.tsx · flow.tsx(공용·메뉴·시간) · api.ts · capture.ts
models/5.4.0/         HEF (git 제외) + SHA256SUMS
scripts/              setup-venv.sh · fetch-hef.sh · bench-npu.py · print-qr.py
deploy/               ecosystem.config.cjs (PM2) · nginx-dave-face.conf · workgate.json
localfiles/           git 제외: DB, admin-token, qr/cards.pdf, logs/, samples/, verification/
```

## 설치

배포판 Hailo apt 패키지는 **설치·삭제 금지** (5.4.0 소스 빌드 드라이버가 지워진다). Python 의존성은 uv 로만.

```bash
bash scripts/setup-venv.sh     # uv venv(/usr/bin/python3, --system-site-packages) + hailort 5.4.0 wheel + gunicorn + segno
bash scripts/fetch-hef.sh      # models/5.4.0/*.hef 다운로드 + SHA-256 검증
pnpm install && pnpm build     # apps/web/dist (index.html + admin.html)
```

PATH 의 `python3` 가 다른 venv(3.11)일 수 있으니 항상 `apps/api/.venv/bin/python` 을 쓴다.

## 실행·배포

```bash
env -i HOME=$HOME USER=$USER LANG=$LANG PATH=$NVM_BIN:/usr/bin:/bin pm2 start deploy/ecosystem.config.cjs --only dave-face
pm2 restart dave-face          # 코드·빌드 변경 반영
pm2 logs dave-face
apps/api/run-web.sh            # 개발용 직접 실행 (127.0.0.1:3011)
```

**NPU 는 한 프로세스만 쓸 수 있다.** dave-face 가 떠 있으면 piWanKanobi·bench 는 `HAILO_OUT_OF_PHYSICAL_DEVICES` 로 실패한다.

```bash
pm2 stop dave-face && pm2 start piwankanobi    # VLM 으로 전환
pm2 stop piwankanobi && pm2 start dave-face    # 얼굴 인식으로 복귀
```

DB 스키마를 바꾸면 `localfiles/dave_face.db` 를 지우고 재시작한다 (마이그레이션 없음). nginx 는
`/etc/nginx/sites-available/default` 의 include 한 줄. WorkGate 갱신: `node ~/works/WorkGate/scripts/py.cjs registry put ./deploy/workgate.json --by <작업자>`.

## 테스트·점검

```bash
pnpm test                      # tsc + vite build + API unittest (NPU 불필요)
curl -s 127.0.0.1:3011/api/health
curl -s -H 'content-type: image/jpeg' --data-binary @face.jpg 127.0.0.1:3011/api/recognize   # 사진 한 장의 유사도

pm2 stop dave-face             # NPU 직접 점검은 서비스를 먼저 내린다
apps/api/.venv/bin/python scripts/bench-npu.py                          # 지연시간
apps/api/.venv/bin/python scripts/bench-npu.py --pair a.jpg b.jpg       # 두 얼굴 코사인
pm2 start dave-face
```

## REST API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/`, `/compare`, `/admin` | 인식 테스트, 사진 비교, 등록(관리자) 페이지 (끝에 `/` 를 붙이면 404) |
| GET | `/api/health`, `/api/version` | 상태 (ready/busy 200, 그 외 503), 버전 |
| POST | `/api/recognize` | 본문 = JPEG → `{status, threshold, face?, scores?, match?, elapsed_ms, npu_ms}` |
| POST | `/api/embed` | 본문 = JPEG → `{status, threshold, face?, embedding?(512), elapsed_ms, npu_ms}` — 저장하지 않는다 |
| POST | `/api/enroll` | 관리자(`X-Admin-Token`). multipart `user_id` + `frame`(JPEG 한 장) → `{user_id, replaced, face}` · 얼굴 없음 422 |
| GET | `/api/users` | 관리자. 등록 목록 |
| DELETE | `/api/users/<id>` | 관리자. 삭제 |

## 주의 사항

- **데모다.** 라이브니스가 없어 사진·화면으로도 일치한다. 문턱 0.5 는 보정 전 값이다.
- `/api/recognize` 는 인증 없이 등록된 ID 와 점수를 돌려준다. 터널로 인터넷에 열려 있으니 얼굴은 동의한 사람만 등록하고,
  관리자 카드·토큰은 아무에게나 주지 않는다.
