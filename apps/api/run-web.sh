#!/usr/bin/env bash
# dave-face 웹 런처. NPU 모델(SCRFD + ArcFace)은 단일 gunicorn worker 에 상주하고 프레임을 한 번에 하나씩 처리한다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
if [[ ! -x "$HERE/.venv/bin/gunicorn" ]]; then
    echo "가상환경이 없습니다: scripts/setup-venv.sh 를 실행하세요" >&2
    exit 2
fi
mkdir -p "$ROOT/localfiles/logs"
# HailoRT 는 기본으로 작업 디렉터리에 hailort.log 를 남긴다 — localfiles/logs 로 모은다.
export HAILORT_LOGGER_PATH="${HAILORT_LOGGER_PATH:-$ROOT/localfiles/logs}"
# --preload 금지: SDK 객체는 워커 프로세스 안에서 만들어야 한다 (pyhailort 가 생성 pid 를 확인한다).
exec "$HERE/.venv/bin/gunicorn" \
    --chdir "$HERE" \
    --bind "127.0.0.1:${DAVE_FACE_PORT:-3011}" \
    --workers 1 --threads 4 --timeout 60 --graceful-timeout 15 \
    --access-logfile - --error-logfile - \
    'dave_face.web:create_app()'
