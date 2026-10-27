import math
import unittest

import numpy as np

from dave_face import pipeline
from dave_face.pipeline import ARCFACE_DST, FacePipeline, Obs, Reject
from tests.test_scrfd import NAMES_10G, empty_heads, put_face


def face_heads(faces):
    """[(stride, row, col, kps(5,2) 원본 좌표, half box)] -> scrfd_10g 원시 헤드 (scale 1 기준)."""
    heads = empty_heads()
    for stride, row, col, kps, half in faces:
        center = np.array([col * stride, row * stride], dtype=np.float32)
        offsets = ((np.asarray(kps) - center) / stride).reshape(-1)
        put_face(heads, NAMES_10G, stride, row, col, 0, 0.9, (half / stride,) * 4, offsets)
    return heads


def template_kps(size, center=(320, 320)):
    """ARCFACE_DST 를 눈 거리 size*35.24 px 로 키워 center 에 둔 5점."""
    return (ARCFACE_DST - 56) * size + np.asarray(center)


class FakeNPU:
    def __init__(self, heads):
        self.heads = heads
        self.crops = []

    def detect_raw(self, rgb640):
        assert rgb640.shape == (640, 640, 3) and rgb640.dtype == np.uint8
        return self.heads

    def embed(self, rgb112):
        assert rgb112.shape == (112, 112, 3) and rgb112.dtype == np.uint8
        self.crops.append(rgb112)
        vector = np.ones(512, dtype=np.float32)
        return vector / np.linalg.norm(vector)


def noise_frame(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (640, 640, 3), dtype=np.uint8)


class UmeyamaTest(unittest.TestCase):
    def test_recovers_known_similarity(self):
        angle = math.radians(10)
        rot = 2.0 * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        src = ARCFACE_DST
        dst = src @ rot.T + np.array([30, -12])
        m = pipeline.umeyama(src, dst)
        np.testing.assert_allclose(m[:, :2], rot, atol=1e-9)
        np.testing.assert_allclose(m[:, 2], (30, -12), atol=1e-9)

    def test_identity_warp_reproduces_image(self):
        frame = noise_frame()[:112, :112].copy()
        crop = pipeline.align(frame, ARCFACE_DST)
        rgb = frame[..., ::-1]
        self.assertLessEqual(int(np.abs(crop.astype(int) - rgb.astype(int)).max()), 1)


class PoseTest(unittest.TestCase):
    def test_template_is_frontal(self):
        eye_dist, roll, yaw = pipeline.pose(ARCFACE_DST)
        self.assertAlmostEqual(eye_dist, 35.24, places=2)
        self.assertLess(abs(roll), 0.5)
        self.assertLess(abs(yaw), 0.01)

    def test_nose_toward_image_right_is_positive_yaw(self):
        kps = template_kps(4)
        kps[2, 0] += 0.3 * 141  # 코를 이미지 오른쪽으로 (= 비미러 프레임에서 본인 왼쪽으로 돌림)
        self.assertGreater(pipeline.pose(kps)[2], 0.25)

    def test_in_plane_rotation_keeps_yaw(self):
        angle = math.radians(20)
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        _, roll, yaw = pipeline.pose(template_kps(4) @ rot.T)
        self.assertAlmostEqual(roll, 20 - 0.32, delta=0.1)
        self.assertLess(abs(yaw), 0.01)


class AnalyzeTest(unittest.TestCase):
    def test_embeds_the_face(self):
        kps = template_kps(3)
        npu = FakeNPU(face_heads([(32, 10, 10, kps, 128)]))
        obs = FacePipeline(npu).analyze(noise_frame())
        self.assertIsInstance(obs, Obs)
        np.testing.assert_allclose(obs.kps, kps, atol=1e-3)
        self.assertAlmostEqual(obs.eye_dist, 3 * 35.24, places=1)
        self.assertGreater(obs.blur, 1000)  # 무작위 잡음 = 아주 선명
        self.assertAlmostEqual(float(np.linalg.norm(obs.emb)), 1.0, places=5)
        self.assertEqual(len(npu.crops), 1)

    def test_no_face(self):
        npu = FakeNPU(empty_heads())
        self.assertEqual(FacePipeline(npu).analyze(noise_frame()), Reject("no_face"))
        self.assertEqual(npu.crops, [])

    def test_small_blurry_and_crowded_faces_are_still_embedded(self):
        # 거르지 않는다: 값만 잰다. 여러 명이면 가장 큰 얼굴.
        two = face_heads([(32, 5, 5, template_kps(3, (160, 160)), 128), (32, 15, 15, template_kps(2, (480, 480)), 120)])
        obs = FacePipeline(FakeNPU(two)).analyze(noise_frame())
        self.assertAlmostEqual(obs.eye_dist, 3 * 35.24, places=1)
        small = FacePipeline(FakeNPU(face_heads([(32, 10, 10, template_kps(1), 128)]))).analyze(noise_frame())
        self.assertAlmostEqual(small.eye_dist, 35.24, places=1)
        flat = np.full((640, 640, 3), 128, dtype=np.uint8)
        self.assertLess(FacePipeline(FakeNPU(face_heads([(32, 10, 10, template_kps(3), 128)]))).analyze(flat).blur, 1)

    def test_blur_ignores_the_black_band_outside_the_frame(self):
        # 얼굴 크롭이 프레임 오른쪽 밖으로 나가는 위치: warpAffine 이 검은 띠를 채운다. 안 빼면 평평한 면이 선명해 보인다.
        edge = template_kps(3, center=(600, 320))
        heads = face_heads([(32, 10, 18, edge, 128)])
        smooth = np.full((640, 640, 3), 128, dtype=np.uint8)
        self.assertLess(FacePipeline(FakeNPU(heads)).analyze(smooth).blur, 1)


if __name__ == "__main__":
    unittest.main()
