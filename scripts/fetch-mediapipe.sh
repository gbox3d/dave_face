#!/usr/bin/env bash
# 얼굴 메시(MediaPipe Face Landmarker, 브라우저 쪽) 자산을 apps/web/public/assets/mediapipe/ 에 둔다.
# Vite 가 public/ 을 dist/ 로 복사하고 API 가 dist/assets/ 를 /assets/ 로 서빙하니, 페이지는 ./assets/mediapipe/… 로 받는다.
#  - wasm/: 로더 4개를 node_modules 의 @mediapipe/tasks-vision 에서 복사한다 (SIMD + nosimd 한 쌍씩 — iOS 16.4 미만은 nosimd).
#  - face_landmarker.task: Google 저장소에서 받아 크기와 SHA-256 을 검증한다. MODEL_SHA256 이 비어 있으면
#    받은 파일의 해시를 찍고 3 으로 끝낸다 — 그 값을 아래에 박고 다시 돌린다.
# 먼저 apps/web 에서 pnpm install 을 해 두어야 한다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB="$HERE/../apps/web"
DEST="$WEB/public/assets/mediapipe"
WASM_SRC="$WEB/node_modules/@mediapipe/tasks-vision/wasm"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SIZE=3758596
MODEL_SHA256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

verified() { echo "$1  $2" | sha256sum -c --quiet - 2>/dev/null; }

if [[ ! -d "$WASM_SRC" ]]; then
    echo "오류: $WASM_SRC 가 없습니다 — 먼저 (cd apps/web && pnpm install) 을 하세요" >&2
    exit 1
fi
mkdir -p "$DEST/wasm"
for name in vision_wasm_internal.js vision_wasm_internal.wasm vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do
    cp "$WASM_SRC/$name" "$DEST/wasm/$name"
    echo "copied  wasm/$name"
done

target="$DEST/face_landmarker.task"
if [[ -n "$MODEL_SHA256" && -f "$target" ]] && verified "$MODEL_SHA256" "$target"; then
    echo "ok      face_landmarker.task"
    exit 0
fi
# 받다 만 .tmp 는 이어 받고, 다 받았지만 해시를 박기 전에 끝난(exit 3) .tmp 는 그대로 검증한다.
if [[ ! -f "$target.tmp" || "$(stat -c %s "$target.tmp")" -lt "$MODEL_SIZE" ]]; then
    echo "download face_landmarker.task"
    curl --fail --location --continue-at - --output "$target.tmp" "$MODEL_URL"
fi
size="$(stat -c %s "$target.tmp")"
if [[ "$size" -ne "$MODEL_SIZE" ]]; then
    echo "오류: face_landmarker.task 크기가 $size 바이트입니다 (기대 $MODEL_SIZE). $target.tmp 를 지우고 다시 돌리세요" >&2
    exit 2
fi
sum="$(sha256sum "$target.tmp" | cut -d' ' -f1)"
if [[ -z "$MODEL_SHA256" ]]; then
    echo "MODEL_SHA256 이 비어 있습니다. 받은 face_landmarker.task 의 SHA-256:" >&2
    echo "  $sum" >&2
    echo "이 값을 $0 의 MODEL_SHA256 에 박고 다시 돌리세요 ($target.tmp 는 남겨 둡니다)" >&2
    exit 3
fi
if [[ "$sum" != "$MODEL_SHA256" ]]; then
    echo "오류: face_landmarker.task SHA-256 이 다릅니다: $sum (기대 $MODEL_SHA256). $target.tmp 를 지웁니다" >&2
    rm -f "$target.tmp"
    exit 2
fi
mv "$target.tmp" "$target"
echo "ok      face_landmarker.task"
