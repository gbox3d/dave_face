#!/usr/bin/env bash
# apps/api/.venv 를 uv 로 만든다. apt 는 쓰지 않는다 (배포판 Hailo 패키지는 5.4.0 드라이버를 지운다).
#  - 시스템 Python 3.13 + --system-site-packages: OpenCV·NumPy·Pillow·Flask 는 시스템 것을 쓴다.
#  - HailoRT 5.4.0 SDK wheel 은 --no-deps: wheel 이 numpy 를 요구해도 시스템 numpy(cv2 가 빌드된 버전)를 지킨다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API="$(cd "$HERE/../apps/api" && pwd)"
DIST=/opt/hailort/5.4.0/dist
PY="$API/.venv/bin/python"

(cd "$DIST" && sha256sum -c SHA256SUMS)
# PATH 의 python3 는 다른 venv 의 3.11 일 수 있어 시스템 인터프리터를 명시한다.
uv venv --allow-existing --python /usr/bin/python3 --system-site-packages "$API/.venv"
uv pip install --python "$PY" --no-deps "$DIST/hailort-5.4.0-cp313-cp313-linux_aarch64.whl"
uv pip install --python "$PY" "gunicorn>=23,<24" "segno>=1.6,<2"
