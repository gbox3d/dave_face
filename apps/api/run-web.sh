#!/usr/bin/env bash
# dave-face 웹 런처 (Jetson). TensorRT 엔진 둘(SCRFD + ArcFace)은 단일 gunicorn worker 에 상주하고 프레임을 한 번에 하나씩 처리한다.
set -euo pipefail
umask 077   # localfiles(DB·관리자 토큰·QR PDF)는 소유자만 읽는다
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
if [[ ! -x "$HERE/.venv/bin/gunicorn" ]]; then
    echo "가상환경이 없습니다: scripts/setup-venv.sh 를 실행하세요" >&2
    exit 2
fi
TRT_VER="$("$HERE/.venv/bin/python" -c 'import tensorrt; print(tensorrt.__version__)')"
for name in scrfd_10g arcface_mobilefacenet; do
    engine="$ROOT/models/trt/$name.fp16.trt$TRT_VER.engine"
    [[ -f "$engine" ]] || { echo "엔진이 없습니다: $engine — scripts/fetch-onnx.sh 와 scripts/build-engines.sh 를 실행하세요" >&2; exit 2; }
done
# GPU 인터락: 4 GB 보드에서 oni-detector(127.0.0.1:9082)와 동시에 돌릴 수 없다 — 먼저 pm2 stop oni-detector.
if ss -H -ltn 'sport = :9082' | grep -q .; then
    echo "oni-detector holds the GPU (:9082 listening) — pm2 stop oni-detector first" >&2
    exit 2
fi
PORT="${DAVE_FACE_PORT:-3011}"
if ss -H -ltn "sport = :$PORT" | grep -q .; then
    echo "127.0.0.1:$PORT 이 이미 열려 있습니다 (다른 dave-face 가 떠 있나?)" >&2
    exit 2
fi
mkdir -p "$ROOT/localfiles/logs"
echo 800 > /proc/self/oom_score_adj 2>/dev/null || true   # 메모리가 바닥나면 WorkGate·pm2 보다 먼저 죽는 쪽이 된다 (gunicorn 이 상속)
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
# --preload 금지: CUDA 컨텍스트는 fork 를 건너지 못한다 — tensorrt import·엔진 로드는 워커 프로세스의 추론 스레드 안에서만 한다.
exec "$HERE/.venv/bin/gunicorn" \
    --chdir "$HERE" \
    --bind "127.0.0.1:$PORT" \
    --workers 1 --threads 4 --timeout 60 --graceful-timeout 15 \
    --access-logfile - --error-logfile - \
    'dave_face.web:create_app()'
