#!/usr/bin/env bash
# apps/api/.venv 를 uv 로 만든다 (Jetson, JetPack 6.0). apt 는 쓰지 않는다.
#  - 시스템 Python 3.10 + --system-site-packages: tensorrt 8.6.2·OpenCV·NumPy·Pillow 는 시스템 것을 쓴다.
#  - cuda-python 12.2.x(cudart 바인딩, import 때 cython 필요)·Flask·gunicorn·segno·tomli 는 venv 에 설치한다.
#    어느 것도 numpy 를 요구하지 않으므로 시스템 numpy 1.21.5(cv2 가 빌드된 버전)는 그대로 남는다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API="$(cd "$HERE/../apps/api" && pwd)"
PY="$API/.venv/bin/python"
UV="$(command -v uv 2>/dev/null || true)"; [[ -x "${UV:-}" ]] || UV="$HOME/.local/bin/uv"   # systemd 유닛 PATH 에는 ~/.local/bin 이 없다
[[ -x "$UV" ]] || { echo "uv 가 없습니다 (~/.local/bin/uv)" >&2; exit 2; }

/usr/bin/python3 -c "import tensorrt" 2>/dev/null || { echo "시스템 python3 에 tensorrt 가 없습니다 (JetPack 6.0 / TensorRT 8.6 필요)" >&2; exit 2; }
"$UV" venv --allow-existing --python /usr/bin/python3 --system-site-packages "$API/.venv"
"$UV" pip install --python "$PY" "cuda-python==12.2.*" cython "Flask>=3.1,<4" "gunicorn>=23,<24" "segno>=1.6,<2" \
    "tomli>=2,<3; python_version < '3.11'"
"$PY" - <<'PYEOF'
import cv2, flask, gunicorn, numpy, segno, tensorrt
from cuda import cudart
err, n = cudart.cudaGetDeviceCount()
assert err == cudart.cudaError_t.cudaSuccess, err
print(f"ok: tensorrt {tensorrt.__version__}, cuda devices {n}, numpy {numpy.__version__} ({numpy.__file__}), cv2 {cv2.__version__}, flask {flask.__version__}")
PYEOF
