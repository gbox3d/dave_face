"""프레임 한 장 -> 얼굴 관측: SCRFD 검출 -> 5점 Umeyama 정렬(112x112) -> ArcFace 임베딩 + 품질·자세 값.

NPU 는 주입받는다 (detect_raw, embed 를 가진 객체). 이 모듈은 hailo_platform 을 import 하지 않는다.
"""

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import scrfd

# InsightFace arcface_dst: 112x112 템플릿의 5점. 순서 = 이미지 기준 왼눈, 오른눈, 코, 왼입꼬리, 오른입꼬리 (SCRFD kps 순서).
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]])


@dataclass(frozen=True)
class Obs:
    bbox: np.ndarray   # (4,) x1 y1 x2 y2, 원본 좌표
    kps: np.ndarray    # (5, 2)
    eye_dist: float    # px, 얼굴 크기 (템플릿의 눈 거리는 35.2 px)
    roll: float        # 도, 눈을 잇는 선의 기울기
    yaw: float         # r = (d_l - d_r) / (d_l + d_r): 정면 ~0, 본인 왼쪽으로 돌리면 + (비미러 프레임)
    blur: float        # 선명도 — 2026-10-27 측정: 선명 780-840, 25px 흔들림 119
    emb: np.ndarray    # (512,) L2 정규화
    npu_ms: float = field(default=0.0, compare=False)  # NPU 호출(검출 + 임베딩)에 걸린 시간


@dataclass(frozen=True)
class Reject:
    reason: str        # no_face
    npu_ms: float = field(default=0.0, compare=False)


def umeyama(src, dst):
    """최소제곱 유사변환(스케일·회전·이동, 반사 없음) src -> dst 의 2x3 행렬. skimage SimilarityTransform 과 같다."""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.array([1.0, -1.0 if np.linalg.det(cov) < 0 else 1.0])
    rotation = u @ np.diag(d) @ vt
    scale = (s * d).sum() / (src_c ** 2).sum(axis=1).mean()
    return np.hstack([scale * rotation, (dst_mean - scale * rotation @ src_mean)[:, None]])


def align(bgr, kps):
    """원본 해상도 BGR 프레임 + 5점 -> 정렬된 112x112 RGB uint8."""
    crop = cv2.warpAffine(bgr, umeyama(kps, ARCFACE_DST), (112, 112), borderValue=0)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def sharpness(bgr, kps, crop):
    """정렬 크롭 회색조의 Laplacian 분산. 프레임 밖(warpAffine 이 채운 검은 띠)과 그 경계 픽셀은 뺀다 —
    안 빼면 검은 띠의 경계만으로 흐린 얼굴이 선명하게 나온다."""
    inside = cv2.warpAffine(np.full(bgr.shape[:2], 255, np.uint8), umeyama(kps, ARCFACE_DST), (112, 112)) == 255
    inside = cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if inside.sum() < 1000:  # 크롭 대부분이 프레임 밖
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F)[inside].var())


def pose(kps):
    """5점 -> (눈 사이 거리 px, roll 도, yaw 비율 r)."""
    left_eye, right_eye, nose, left_mouth, right_mouth = kps
    eye = right_eye - left_eye
    d_l = math.dist(nose, (left_eye + left_mouth) / 2)
    d_r = math.dist(nose, (right_eye + right_mouth) / 2)
    return (float(math.hypot(*eye)), math.degrees(math.atan2(eye[1], eye[0])),
            (d_l - d_r) / max(d_l + d_r, 1e-9))


class FacePipeline:
    def __init__(self, npu):
        self.npu = npu

    def analyze(self, bgr):
        """BGR 프레임 -> Obs | Reject("no_face"). 가장 큰 얼굴 하나만 본다.
        크기·선명도·자세는 값만 재고 거르지 않는다 — 데모라 얼굴이 보이면 항상 임베딩한다."""
        canvas, scale = scrfd.letterbox(bgr)
        started = time.perf_counter()
        raw = self.npu.detect_raw(canvas)
        npu_ms = (time.perf_counter() - started) * 1000
        faces = scrfd.decode(raw, scale)
        if not faces:
            return Reject("no_face", npu_ms)
        bbox, _, kps = faces[0]
        eye_dist, roll, yaw = pose(kps)
        crop = align(bgr, kps)
        started = time.perf_counter()
        emb = self.npu.embed(crop)
        npu_ms += (time.perf_counter() - started) * 1000
        return Obs(bbox, kps, eye_dist, roll, yaw, sharpness(bgr, kps, crop), emb, npu_ms)
