"""Jetson(Orin) 가속기 백엔드: TensorRT 8.6 Python API + cuda-python 12.2 cudart. tensorrt / cuda 를 import 하는 유일한 모듈이다.

파일 이름과 FaceNPU 계약(detect_raw, embed, det/rec.input_shape·outputs, close)은 Hailo 판과 같아 pipeline.py·web.py·bench 가 그대로 쓴다
(이름의 NPU 는 '가속기' 라는 뜻으로 남겨 둔다). 엔진은 scripts/build-engines.sh 가 trtexec 로 이 보드에서 만든다 — TRT 버전·GPU 전용, git 제외.
HEF 안에 있던 입력 정규화는 여기(호스트)서 하고, SCRFD 점수 시그모이드는 ONNX 그래프 안에 있다:
  SCRFD   RGB 640x640 uint8 -> (x-127.5)/128   NCHW float32 -> 출력 9개 (1, N, C) -> (n, n, 2C)  (scrfd._heads 가 shape 으로 고른다)
  ArcFace RGB 112x112 uint8 -> (x-127.5)/127.5 NCHW float32 -> (1, 512) -> L2 정규화
CUDA 컨텍스트는 trt.Runtime 을 만들 때 생기고 fork 를 건너지 못한다 — 워커 프로세스의 추론 스레드 안에서만 만든다 (gunicorn --preload 금지).
"""

import logging
import math
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart  # cuda-python 12.2: 패키지 'cuda', 모듈 cuda.cudart (libcuda 위에 구현, libcudart 불필요)

from .scrfd import ANCHORS, INPUT_SIZE, STRIDES

LOG = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "trt"
# 엔진은 TRT 버전·GPU 에 묶인다 — 이름에 버전을 넣어 JetPack 을 올리면 '엔진 없음' 으로 드러나게 한다.
DET_ENGINE = MODEL_DIR / f"scrfd_10g.fp16.trt{trt.__version__}.engine"
REC_ENGINE = MODEL_DIR / f"arcface_mobilefacenet.fp16.trt{trt.__version__}.engine"
H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
GRIDS = tuple(INPUT_SIZE // s for s in STRIDES)  # (80, 40, 20)
EMBEDDING = 512


def _cuda(result, what):
    """cudart 는 (cudaError_t, *values) 를 돌려준다 — 오류면 예외, 아니면 값(하나면 벗겨서)."""
    err, *values = result
    if err != cudart.cudaError_t.cudaSuccess:
        _, text = cudart.cudaGetErrorString(err)  # (cudaSuccess, bytes)
        raise RuntimeError(f"{what}: {err.name} ({text.decode()})")
    return values[0] if len(values) == 1 else values


def _hwc(name, array):
    """SCRFD 출력 (1, N, C) 또는 (N, C), N = n*n*ANCHORS 가 (h, w, anchor) 순 -> HEF 와 같은 (n, n, ANCHORS*C)."""
    rows = array.reshape(-1, array.shape[-1])
    n = math.isqrt(rows.shape[0] // ANCHORS)
    if n * n * ANCHORS != rows.shape[0] or n not in GRIDS:
        raise RuntimeError(f"{name}: unexpected SCRFD output shape {array.shape}")
    return rows.reshape(n, n, ANCHORS * rows.shape[1])


class _Logger(trt.ILogger):
    """TensorRT 메시지를 logging 으로 넘기고 마지막 오류를 기억한다 (deserialize 는 실패해도 None 만 돌려준다)."""

    def __init__(self):
        super().__init__()
        self.last_error = ""

    def log(self, severity, msg):
        if severity in (trt.ILogger.Severity.INTERNAL_ERROR, trt.ILogger.Severity.ERROR):
            self.last_error = msg
            LOG.error("TensorRT: %s", msg)
        elif severity == trt.ILogger.Severity.WARNING:
            LOG.warning("TensorRT: %s", msg)


class _Model:
    """엔진 하나: 입력 1개(1x3xHxW float32) + 출력 N개. 디바이스 버퍼는 한 번 할당해 재사용한다."""

    def __init__(self, runtime, logger, path, mean, std, stream):
        self._logger, self._mean, self._std, self._stream = logger, np.float32(mean), np.float32(std), stream
        self._bufs = {}  # name -> (device ptr, host array)
        self.context = self.engine = None
        try:
            self._load(runtime, Path(path))
        except BaseException:
            self.close()
            raise

    def _load(self, runtime, path):
        if not path.is_file():
            raise RuntimeError(f"engine missing: {path} — run scripts/build-engines.sh")
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:  # TRT 버전·GPU 불일치·손상 — 이유는 로거에만 온다
            raise RuntimeError(f"{path.name}: deserialize failed ({self._logger.last_error}) — run scripts/build-engines.sh")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"{path.name}: create_execution_context failed ({self._logger.last_error})")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        if len(inputs) != 1:
            raise RuntimeError(f"{path.name}: expected one input, got {inputs}")
        self._in = inputs[0]
        shape = tuple(self.engine.get_tensor_shape(self._in))
        if -1 in shape:  # 동적 축(배치) — 프로파일의 opt 모양으로 고정한다
            shape = tuple(self.engine.get_tensor_profile_shape(self._in, 0)[1])
            if not self.context.set_input_shape(self._in, shape):
                raise RuntimeError(f"{path.name}: set_input_shape{shape} rejected")
            if self.context.infer_shapes():
                raise RuntimeError(f"{path.name}: output shapes unresolved")
        if len(shape) != 4 or shape[:2] != (1, 3):
            raise RuntimeError(f"{path.name}: expected 1x3xHxW input, got {shape}")
        self.input_shape = (shape[2], shape[3], 3)  # HWC — pipeline·bench 계약
        for name in names:
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            tshape = tuple(self.context.get_tensor_shape(name))
            if dtype != np.float32 or -1 in tshape:  # --fp16 으로 빌드해도 I/O 는 fp32:chw 가 기본이다
                raise RuntimeError(f"{path.name}: {name} must be a static float32 tensor, got {dtype} {tshape}")
            host = np.empty(tshape, dtype=np.float32)
            dev = _cuda(cudart.cudaMalloc(host.nbytes), f"cudaMalloc {name}")
            self._bufs[name] = (dev, host)
            if not self.context.set_tensor_address(name, dev):
                raise RuntimeError(f"{path.name}: set_tensor_address({name}) rejected")
        self.outputs = {name: host.shape for name, (_, host) in self._bufs.items() if name != self._in}

    def infer(self, image):
        """uint8 HWC 이미지 하나 -> {출력 이름: float32 배열(엔진 모양 그대로, 복사본)}."""
        if image.shape != self.input_shape or image.dtype != np.uint8:
            raise ValueError(f"input {image.shape} {image.dtype} != {self.input_shape} uint8")
        x = np.ascontiguousarray(((image.astype(np.float32) - self._mean) / self._std).transpose(2, 0, 1)[None])
        _cuda(cudart.cudaMemcpyAsync(self._bufs[self._in][0], x.ctypes.data, x.nbytes, H2D, self._stream), "H2D")
        if not self.context.execute_async_v3(int(self._stream)):  # 핸들 값은 int(stream) — getPtr() 은 포인터의 포인터
            raise RuntimeError(f"execute_async_v3 failed ({self._logger.last_error})")
        for name, (dev, host) in self._bufs.items():
            if name != self._in:
                _cuda(cudart.cudaMemcpyAsync(host.ctypes.data, dev, host.nbytes, D2H, self._stream), "D2H")
        _cuda(cudart.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize")  # 이 뒤에야 host 버퍼가 유효하다
        return {name: host.copy() for name, (_, host) in self._bufs.items() if name != self._in}

    def close(self):
        for dev, _ in self._bufs.values():
            cudart.cudaFree(dev)
        self._bufs = {}
        self.context = self.engine = None  # 컨텍스트 -> 엔진 순으로 참조를 놓는다


class FaceNPU:
    """런타임 하나 + 스트림 하나 + 엔진 둘. 생성·사용·해제를 모두 추론 스레드(워커 프로세스 안, fork 뒤)에서 한다."""

    def __init__(self, det_engine=DET_ENGINE, rec_engine=REC_ENGINE):
        self.det = self.rec = self._stream = None
        self._logger = _Logger()
        trt.init_libnvinfer_plugins(self._logger, "")  # 플러그인 없는 CNN 에는 불필요하지만 무해하다
        self._runtime = trt.Runtime(self._logger)  # 여기서 CUDA 컨텍스트가 생긴다
        try:
            self._stream = _cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")
            self.det = _Model(self._runtime, self._logger, det_engine, 127.5, 128.0, self._stream)
            self.rec = _Model(self._runtime, self._logger, rec_engine, 127.5, 127.5, self._stream)
            self._check()
        except BaseException:
            self.close()
            raise

    def _check(self):
        """엔진 계약 확인 + 워밍업 — CUDA 지연 로딩·첫 커널 적재를 첫 요청이 아니라 여기서 치른다."""
        heads = self.detect_raw(np.zeros(self.det.input_shape, dtype=np.uint8))
        shapes = sorted((a.shape[0], a.shape[2]) for a in heads.values())
        expected = sorted((n, ANCHORS * c) for n in GRIDS for c in (1, 4, 10))
        if shapes != expected:
            raise RuntimeError(f"SCRFD engine heads {shapes} != {expected}")
        for name, a in heads.items():
            if a.shape[2] == ANCHORS and not (0.0 <= float(a.min()) and float(a.max()) <= 1.0):
                raise RuntimeError(f"{name}: SCRFD scores are not in [0, 1] — the engine must come from the full ONNX (with Sigmoid)")
        if len(self.rec.outputs) != 1 or sum(int(np.prod(s)) for s in self.rec.outputs.values()) != EMBEDDING:
            raise RuntimeError(f"ArcFace engine outputs {self.rec.outputs} != one {EMBEDDING}-D vector")
        self.embed(np.zeros(self.rec.input_shape, dtype=np.uint8))

    def detect_raw(self, rgb640):
        """640x640x3 RGB uint8 -> SCRFD 원시 헤드 {출력 이름: float32 (n, n, 2C)}, 점수는 [0, 1], bbox·kps 는 스트라이드 단위."""
        return {name: _hwc(name, a) for name, a in self.det.infer(rgb640).items()}

    def embed(self, rgb112):
        """정렬된 112x112x3 RGB uint8 -> L2 정규화된 512-D float32."""
        (vector,) = self.rec.infer(rgb112).values()
        vector = vector.reshape(-1)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def close(self):
        for model in (self.det, self.rec):
            if model is not None:
                model.close()
        self.det = self.rec = None
        if self._stream is not None:
            cudart.cudaStreamDestroy(self._stream)
            self._stream = None
        self._runtime = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
