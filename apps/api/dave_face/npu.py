"""Hailo-10H NPU: VDevice 하나에 SCRFD(검출)와 ArcFace(임베딩)를 올린다.

hailo_platform 을 import 하는 유일한 모듈이다. 10H 는 HRPC 클라이언트라 InferModel API 만 동작한다
(VDevice.configure / InferVStreams 는 HAILO_NOT_IMPLEMENTED). 입출력은 HEF 그대로 UINT8 로 두고,
출력은 HEF 의 양자화 값으로 호스트에서 역양자화한다: float = (q - zp) * scale.
"""

import contextlib
from pathlib import Path

import numpy as np
from hailo_platform import FormatType, VDevice

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "5.4.0"
DET_HEF = MODEL_DIR / "scrfd_10g.hef"
REC_HEF = MODEL_DIR / "arcface_mobilefacenet.hef"
TIMEOUT_MS = 2000


class _Model:
    """configure 된 InferModel 하나. 입출력 shape 과 양자화 값은 HEF 에서 읽는다."""

    def __init__(self, vdevice, stack, hef_path):
        infer_model = vdevice.create_infer_model(str(hef_path))
        infer_model.set_batch_size(1)
        infer_model.input().set_format_type(FormatType.UINT8)
        self.input_shape = tuple(infer_model.input().shape)
        self.outputs = {}  # name -> (shape, zero point, scale)
        for name in infer_model.output_names:
            stream = infer_model.output(name)
            stream.set_format_type(FormatType.UINT8)
            quant = stream.quant_infos[0]
            self.outputs[name] = (tuple(stream.shape), float(quant.qp_zp), float(quant.qp_scale))
        # 스케줄러가 켜져 있으므로 activate() 는 부르지 않는다.
        self._configured = stack.enter_context(infer_model.configure())

    def infer(self, image):
        """uint8 이미지 하나 -> {출력 이름: 역양자화된 float32 배열}."""
        if image.shape != self.input_shape:
            raise ValueError(f"input shape {image.shape} != {self.input_shape}")
        buffers = {name: np.empty(shape, dtype=np.uint8) for name, (shape, _, _) in self.outputs.items()}
        bindings = self._configured.create_bindings(output_buffers=buffers)
        # SDK 가 쓰기 가능한 C-연속 버퍼를 요구하므로 항상 새 배열로 넘긴다.
        bindings.input().set_buffer(np.array(image, dtype=np.uint8, order="C"))
        self._configured.run([bindings], TIMEOUT_MS)
        return {name: (buffers[name].astype(np.float32) - zp) * scale
                for name, (_, zp, scale) in self.outputs.items()}


class FaceNPU:
    """VDevice 하나 + InferModel 두 개. 프로세스당 하나만 열고, 끝나면 close() 로 장치를 놓는다."""

    def __init__(self, det_hef=DET_HEF, rec_hef=REC_HEF):
        self._stack = contextlib.ExitStack()
        try:
            # 기본값: ROUND_ROBIN 스케줄러, group_id 'UNIQUE'(NPU 독점), multi_process_service False.
            vdevice = self._stack.enter_context(VDevice(VDevice.create_params()))
            self.det = _Model(vdevice, self._stack, det_hef)
            self.rec = _Model(vdevice, self._stack, rec_hef)
        except BaseException:
            self._stack.close()
            raise

    def detect_raw(self, rgb640):
        """640x640x3 RGB uint8 -> SCRFD 원시 헤드 {출력 이름: float32 (H, W, C)}."""
        return self.det.infer(rgb640)

    def embed(self, rgb112):
        """정렬된 112x112x3 RGB uint8 -> L2 정규화된 512-D float32."""
        (vector,) = self.rec.infer(rgb112).values()
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
