import unittest

import numpy as np

from dave_face import scrfd

# scrfd_10g 의 레이어 이름 (스트라이드 8/16/32 x 점수/bbox/kps)
NAMES_10G = {(80, 2): "conv41", (80, 8): "conv42", (80, 20): "conv43",
             (40, 2): "conv49", (40, 8): "conv50", (40, 20): "conv51",
             (20, 2): "conv56", (20, 8): "conv57", (20, 20): "conv58"}
# scrfd_2.5g 는 같은 헤드가 다른 이름을 갖는다
NAMES_25G = {(80, 2): "conv42", (80, 8): "conv43", (80, 20): "conv44",
             (40, 2): "conv49", (40, 8): "conv50", (40, 20): "conv51",
             (20, 2): "conv55", (20, 8): "conv56", (20, 20): "conv57"}


def empty_heads(names=NAMES_10G):
    return {f"scrfd/{name}": np.zeros((n, n, c), dtype=np.float32) for (n, c), name in names.items()}


def put_face(heads, names, stride, row, col, anchor, score, dist, offsets):
    n = 640 // stride
    heads[f"scrfd/{names[(n, 2)]}"][row, col, anchor] = score
    heads[f"scrfd/{names[(n, 8)]}"][row, col, anchor * 4:anchor * 4 + 4] = dist
    heads[f"scrfd/{names[(n, 20)]}"][row, col, anchor * 10:anchor * 10 + 10] = offsets


class DecodeTest(unittest.TestCase):
    def test_single_anchor_closed_form(self):
        heads = empty_heads()
        offsets = np.arange(10, dtype=np.float32) / 10
        put_face(heads, NAMES_10G, 8, row=10, col=20, anchor=1, score=0.9, dist=(1, 1, 1, 1), offsets=offsets)
        faces = scrfd.decode(heads, scale=1.0)
        self.assertEqual(len(faces), 1)
        bbox, score, kps = faces[0]
        np.testing.assert_allclose(bbox, (152, 72, 168, 88))
        self.assertAlmostEqual(score, 0.9, places=6)
        expected = np.array([160, 80], dtype=np.float32) + offsets.reshape(5, 2) * 8
        np.testing.assert_allclose(kps, expected, rtol=1e-6)

    def test_scale_maps_back_to_original_pixels(self):
        heads = empty_heads()
        put_face(heads, NAMES_10G, 16, row=5, col=5, anchor=0, score=0.8, dist=(2, 2, 2, 2), offsets=np.zeros(10))
        (bbox, _, kps), = scrfd.decode(heads, scale=1 / 3)
        np.testing.assert_allclose(bbox, np.array([48, 48, 112, 112]) * 3, rtol=1e-6)
        np.testing.assert_allclose(kps, np.full((5, 2), 80 * 3), rtol=1e-6)

    def test_threshold_drops_low_scores(self):
        heads = empty_heads()
        put_face(heads, NAMES_10G, 32, row=3, col=3, anchor=0, score=0.49, dist=(1, 1, 1, 1), offsets=np.zeros(10))
        self.assertEqual(scrfd.decode(heads, scale=1.0), [])

    def test_nms_keeps_higher_score_of_overlapping_pair(self):
        heads = empty_heads()
        # 같은 셀의 두 앵커: 거의 같은 박스 (IoU ~0.8)
        put_face(heads, NAMES_10G, 8, row=40, col=40, anchor=0, score=0.7, dist=(4, 4, 4, 4), offsets=np.zeros(10))
        put_face(heads, NAMES_10G, 8, row=40, col=40, anchor=1, score=0.95, dist=(4, 4, 3, 3), offsets=np.zeros(10))
        faces = scrfd.decode(heads, scale=1.0)
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0][1], 0.95, places=6)

    def test_separate_faces_sorted_by_area(self):
        heads = empty_heads()
        put_face(heads, NAMES_10G, 8, row=10, col=10, anchor=0, score=0.9, dist=(1, 1, 1, 1), offsets=np.zeros(10))
        put_face(heads, NAMES_10G, 32, row=15, col=15, anchor=0, score=0.6, dist=(2, 2, 2, 2), offsets=np.zeros(10))
        faces = scrfd.decode(heads, scale=1.0)
        self.assertEqual(len(faces), 2)
        self.assertAlmostEqual(faces[0][1], 0.6, places=6)  # 128x128 박스가 16x16 보다 먼저

    def test_heads_matched_by_shape_not_name(self):
        args = dict(stride=8, row=10, col=20, anchor=1, score=0.9, dist=(1, 1, 1, 1),
                    offsets=np.arange(10, dtype=np.float32))
        a, b = empty_heads(NAMES_10G), empty_heads(NAMES_25G)
        put_face(a, NAMES_10G, **args)
        put_face(b, NAMES_25G, **args)
        (box_a, score_a, kps_a), = scrfd.decode(a, scale=1.0)
        (box_b, score_b, kps_b), = scrfd.decode(b, scale=1.0)
        np.testing.assert_array_equal(box_a, box_b)
        np.testing.assert_array_equal(kps_a, kps_b)
        self.assertEqual(score_a, score_b)


class LetterboxTest(unittest.TestCase):
    def test_landscape_pads_bottom(self):
        frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        canvas, scale = scrfd.letterbox(frame)
        self.assertEqual(canvas.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 1 / 3)
        self.assertTrue((canvas[:360] == 200).all())
        self.assertTrue((canvas[360:] == 0).all())

    def test_portrait_pads_right(self):
        frame = np.full((1920, 1080, 3), 200, dtype=np.uint8)
        canvas, scale = scrfd.letterbox(frame)
        self.assertAlmostEqual(scale, 1 / 3)
        self.assertTrue((canvas[:, :360] == 200).all())
        self.assertTrue((canvas[:, 360:] == 0).all())

    def test_extreme_aspect_ratio_does_not_crash(self):
        canvas, scale = scrfd.letterbox(np.full((2, 1920, 3), 200, dtype=np.uint8))
        self.assertEqual(canvas.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 1 / 3)
        self.assertTrue((canvas[0] == 200).all())

    def test_converts_bgr_to_rgb(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[..., 0] = 255  # OpenCV BGR 의 파랑
        canvas, scale = scrfd.letterbox(frame)
        self.assertEqual(scale, 1.0)
        self.assertTrue((canvas[:480, :, 2] == 255).all())
        self.assertTrue((canvas[:480, :, 0] == 0).all())


if __name__ == "__main__":
    unittest.main()
