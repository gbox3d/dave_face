# Memo

## 목차

- [제품 기준선](#제품-기준선)
- [기본 설정값](#기본-설정값)
- [런타임 구조 메모](#런타임-구조-메모)
- [동작 규칙](#동작-규칙)
- [반복 금지](#반복-금지)

## 제품 기준선

- 보드: Raspberry Pi 5 Model B 8 GiB (Debian 13, 커널 6.12 rpt) + Hailo-10H AI HAT+ 2. NPU 메모리는 HAT 전용 8 GB(제품 사양 — `hailortcli fw-control identify` 로는 안 나옴)로 Pi RAM 과 별개, PCIe Gen3 x1 로 연결.
- HailoRT/FW 5.4.0 소스 빌드 `/opt/hailort/5.4.0` (wheel `dist/hailort-5.4.0-cp313-cp313-linux_aarch64.whl`). 배포판 Hailo apt 패키지는 설치·삭제 금지 (5.4.0 드라이버·펌웨어를 지운다).
- 모델: Hailo Model Zoo v5.4.0 hailo10h — `scrfd_10g.hef`(4,620,288 B), `arcface_mobilefacenet.hef`(2,666,496 B). 해시는 `models/5.4.0/SHA256SUMS`.
- Python 3.13.5(시스템) + cv2 4.10 · numpy 2.2.4 · Flask 3.1.1 · gunicorn 23 · segno 1.6.6. Node 22 + pnpm 11.1, React 19.3 · Vite 8.3 · TS 7 · qrcode-generator 2.0.4.
- 실측 (2026-10-27, nginx 경유 20회 중앙값): 서버 67 ms, NPU 53 ms (단독 벤치: SCRFD 41 ms + ArcFace 6.7 ms). dave-face 워커 메모리 약 200 MB.
- 공개 샘플 유사도: 같은 사람 다른 해상도 0.980, 같은 사진 50%·35% 축소(눈 사이 36·25 px) 0.946·0.903, 다른 사람 −0.02~0.14.

## 기본 설정값

| 값 | 위치 | 설명 |
|---|---|---|
| 포트 3011 | `run-web.sh`, `deploy/*` | 127.0.0.1 바인드. nginx `/dave-face/` → `127.0.0.1:3011/` |
| `MATCH_THRESHOLD = 0.5` | `web.py` | 코사인이 이 이상이면 "일치". 보정 전 값이라 화면에 그대로 보여 준다 |
| JPEG ≤ 1 MiB, 긴 변 240–1920, 짧은 변 ≥ 120 | `web.py` | 아니면 413/422 |
| 캡처 긴 변 1280, 품질 0.85 | `capture.ts` | 업로드 픽셀은 비미러, 미리보기만 CSS 미러 |
| SCRFD 점수 0.5, NMS 0.4 | `scrfd.py` | 가장 큰 얼굴 하나만 쓴다 |
| 엔진 타임아웃 5 s, 슬롯 대기 1 s | `web.py` `InferenceEngine` | 붐비면 503 busy → 클라이언트가 200 ms 뒤 재시도 |
| 관리자 토큰 | `localfiles/admin-token` (0600, 첫 실행 때 생성) | 접속 QR 은 `/admin#t=<토큰>` (URL 조각 → sessionStorage) |
| 터널 주소 | `~/works/baro_tunnel/localfiles/keeper-out.log` 의 마지막 trycloudflare 호스트 | 재시작마다 바뀐다. 문서·채팅에 적지 않는다 |
| 프린터 | `ippfind` 첫 결과 (FUJI XEROX ApeosPort C2060, `ipp://FX4ff441.local:631/ipp/print`) | CUPS 큐 없이 `ipptool … print-job.test`. 한글 폰트가 없어 카드 글자는 영문 |

## 런타임 구조 메모

- 요청 흐름: 폰(HTTPS 터널) → nginx `/dave-face/` → gunicorn 워커(스레드 4) → `InferenceEngine` 큐(크기 1) → 워커 스레드 하나가 `FacePipeline.analyze` → 응답(`elapsed_ms`, `npu_ms`).
- NPU 는 `group_id='UNIQUE'` 로 프로세스 독점. dave-face 가 떠 있으면 piWanKanobi·bench 는 `HAILO_OUT_OF_PHYSICAL_DEVICES(74)`. `hailortcli monitor`(WorkGate 대시보드)는 공존한다.
- 10H 는 HRPC 클라이언트: `create_infer_model` → `configure()` → `create_bindings(output_buffers=…)` → `run([bindings], 2000)` 만 동작. `VDevice.configure`/`InferVStreams` 는 `HAILO_NOT_IMPLEMENTED`, `activate()` 는 부르지 않는다. 출력은 UINT8 → `(q − zp) × scale` (`quant_infos` 는 리스트).
- SCRFD: 입력 raw RGB 640×640 (정규화·시그모이드는 HEF 내장), 출력 9개를 (H,C) 로 매칭, 셀당 앵커 2, 중심 (col·s, row·s) 에 +0.5 없음. ArcFace: RGB 112×112, 출력 `fc1` 512 → L2.
- 사진 비교(`/compare`)는 서버에 아무것도 남기지 않는다: `/api/embed` 가 임베딩을 주고 브라우저가 내적을 잰다.
- PM2: 앱 `dave-face`(fork, 1 instance). `pm2 start` 는 호출 셸 env 를 복사하므로 `env -i …` 로 띄운다. `pm2 save` 된 정지 앱은 재부팅 뒤에도 정지로 복원된다.
- WorkGate 카드 "Dave Face" (열기 `/dave-face/`, 링크 사진 비교·등록). 갱신: `node ~/works/WorkGate/scripts/py.cjs registry put ./deploy/workgate.json --by …`.

## 동작 규칙

- 데모다: 인증·라이브니스·시도 제한이 없다. `/api/recognize`·`/api/embed` 는 무인증(등록된 ID 와 점수를 돌려준다). 등록·목록·삭제만 관리자 토큰(`X-Admin-Token`, 빈 토큰이면 거부).
- 얼굴이 하나라도 보이면 항상 임베딩한다(가장 큰 얼굴). 크기·선명도·yaw/roll 은 값만 보여 준다.
- 등록은 사진 한 장, 같은 ID 는 교체. DB 스키마를 바꾸면 마이그레이션 없이 파일을 지우고 재시작한다 (지우기 전에 행 수를 확인하고, 등록이 있으면 옮긴다).
- 사진·크롭은 저장하지 않는다. 임베딩만 `localfiles/dave_face.db`(0600). 얼굴은 동의한 사람만 등록한다.

## 반복 금지

- 데모 요청에 요청하지 않은 보안 층을 넣지 말 것: 1:1 ID 로그인·고개 돌리기 챌린지·8장 등록·잠금·품질 거름을 넣었다가, 폰 시도 4번이 모두 챌린지 타임아웃으로 끝나 유사도가 한 번도 안 나왔다 (2026-10-27). 핵심 경로(찍고 비교)를 먼저 만들고, 수치는 그대로 보이게.
- 실기기에서 검증 안 된 문턱을 핵심 경로 앞에 두지 말 것 (yaw 비율 0.25 문턱이 폰에서 넘어가지 않았다).
- `pm2 delete piwankanobi` 금지 — 계속 쓰는 서비스다. NPU 를 넘길 때는 `pm2 stop` 만.
- `pm2 … --no-autostart` 금지 (`autostart:false` 가 남아 나중에 `pm2 start` 해도 안 켜진다).
- Vite 엔트리를 컴포넌트와 대소문자만 다른 이름으로 만들지 말 것 (`admin.tsx` 와 `Admin.tsx` → TS1149). 엔트리는 `*-main.tsx`.
- 페이지 경로 끝에 `/` 를 붙이지 말 것 (`/admin/` 이면 상대 `./assets/`·`api` 가 깨진다 — 그래서 404 로 둔다).
- 백그라운드 점검 스크립트는 `sys.stdout.reconfigure(line_buffering=True)` — 블록 버퍼링 때문에 NPU 독점 테스트를 잘못 판정한 적이 있다.
- `HAILORT_LOGGER_PATH` 를 정하지 않으면 작업 디렉터리에 `hailort.log` 가 생긴다.
- nginx reload 는 비동기라 직후 첫 요청이 옛 워커로 갈 수 있다 — 한 번 더 확인한다.
- iOS Safari 는 화면을 넘길 때 진행 중인 fetch 를 끊는다("Load failed") — 뒤로 가기 캐시 복원(`pageshow` persisted) 때 오류를 지운다.
- 처리 속도(fps)는 마지막 프레임 시각으로 잴 것 — 렌더 시각으로 재면 정지 후에도 숫자가 변한다.
