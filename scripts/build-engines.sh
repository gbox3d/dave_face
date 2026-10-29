#!/usr/bin/env bash
# models/onnx/*.onnx -> models/trt/<name>.fp16.trt<ver>.engine (TensorRT 8.6, FP16 연산, 배치 1, I/O 는 fp32).
# 엔진은 TRT 버전·GPU 전용 — 이 보드에서 만들고 커밋하지 않는다 (models/trt/ 는 git 제외).
# 4 GB 보드: GPU 를 쓰는 서비스를 먼저 내리고(pm2 stop oni-detector), 격리 유닛에서 돌린다:
#   systemd-run --user --unit=dave-face-build --collect --same-dir -p OOMPolicy=continue -p MemoryMax=2500M \
#       -- nice -n 10 choom -n 1000 bash scripts/build-engines.sh
# 실패하면 <engine>.tmp 가 남을 수 있다 — 무해하다(git 제외, 다음 빌드가 덮어쓴다).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
ONNX="$ROOT/models/onnx"; OUT="$ROOT/models/trt"; LOGS="$ROOT/localfiles/logs"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
MIN_AVAILABLE_KB="${MIN_AVAILABLE_KB:-1500000}"   # 빌더 라이브러리 고정비(약 1–2 GB) 때문에 가용 메모리 게이트
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
TRT_VER="$(/usr/bin/python3 -c 'import tensorrt; print(tensorrt.__version__)')"
mkdir -p "$OUT" "$LOGS"
(cd "$ONNX" && sha256sum -c --quiet SHA256SUMS)
available_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
if (( available_kb < MIN_AVAILABLE_KB )); then
    echo "MemAvailable ${available_kb} kB < ${MIN_AVAILABLE_KB} kB — GPU 를 쓰는 서비스를 내리고(pm2 stop oni-detector) 다시 실행하세요" >&2
    ps -eo rss,comm --sort=-rss | head -8 >&2
    exit 2
fi

build() {   # build <onnx 파일> <엔진 이름>
    local onnx="$ONNX/$1" engine="$OUT/$2.fp16.trt${TRT_VER}.engine" log="$LOGS/trt-build-$2.log"
    if [[ -f "$engine" ]]; then echo "ok      $(basename "$engine")"; return; fi
    echo "build   $2 <- $1 (log: $log, MemAvailable $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB)"
    local started=$SECONDS
    # shape 플래그는 주지 않는다: 정적 모델에 주면 trtexec 가 에러로 끝나고, 동적 축(배치)은 trtexec 가 1 로 대체한다(경고만).
    # --tacticSources: cuDNN/cuBLAS 를 빌드·엔진에서 뺀다 (8.6 기본값이기도 하다) — 워커가 그 라이브러리를 적재하지 않게.
    if ! "$TRTEXEC" --onnx="$onnx" --saveEngine="$engine.tmp" --fp16 \
            --memPoolSize=workspace:256 --maxAuxStreams=0 --tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT \
            --builderOptimizationLevel=3 --timingCacheFile="$OUT/timing.cache" \
            --skipInference --verbose --profilingVerbosity=detailed --exportLayerInfo="$LOGS/$2.layers.json" \
            >"$log" 2>&1; then
        tail -30 "$log" >&2
        echo "trtexec 실패 ($2) — OOM 이면 journalctl -k -b | grep -iE 'oom-kill|killed process' 로 확인" >&2
        exit 1
    fi
    mv -f "$engine.tmp" "$engine"
    echo "        $(( SECONDS - started )) s, $(stat -c %s "$engine") bytes"
    grep -E 'Adding network input|as output|Engine built in|Total Host Persistent|Total Device Persistent|Total Activation Memory|Total Weights Memory|Peak memory' "$log" | sed 's/^/        /' || true
    echo "        Sigmoid nodes: $(grep -c '\[Sigmoid\]' "$log" || true)"
}
build mbf.onnx arcface_mobilefacenet          # 작은 것부터 (도구 체인 확인)
build scrfd_10g_bnkps.onnx scrfd_10g
ls -la "$OUT"
