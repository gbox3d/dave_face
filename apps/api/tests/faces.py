"""테스트용 가짜 관측: 템플릿 5점을 키운 얼굴과 서로 가깝거나 먼 임베딩. NPU 없이 쓴다."""

import numpy as np

from dave_face.pipeline import ARCFACE_DST, Obs, pose

EYE = 35.24 * 4  # 템플릿을 4배로 키운 얼굴의 눈 거리 (~141 px)


def unit(v):
    return v / np.linalg.norm(v)


GALLERY = unit(np.random.default_rng(1).normal(size=512))


def same_person(seed, base=GALLERY, noise=0.2):
    """base 와 코사인 ~0.98 인 임베딩 (노이즈 벡터 노름 ~noise)."""
    return unit(base + noise * np.random.default_rng(seed).normal(size=512) / np.sqrt(512)).astype(np.float32)


def stranger(seed=42):
    """GALLERY 와 거의 직교하는 무작위 단위 벡터 (512-D 에서 코사인 ~N(0, 0.044))."""
    return unit(np.random.default_rng(seed).normal(size=512)).astype(np.float32)


def obs(kps, emb):
    eye_dist, roll, yaw = pose(kps)
    return Obs(np.zeros(4), kps, eye_dist, roll, yaw, 500.0, emb)


def frontal(seed=0):
    return obs((ARCFACE_DST - 56) * 4 + 320, same_person(seed))

