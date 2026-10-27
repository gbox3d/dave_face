"""SCRFD 전처리·후처리 (Model Zoo v5.4.0 hailo10h HEF 계약). numpy·OpenCV 만 쓴다.

- 입력: raw RGB uint8 640x640. 정규화 (x-127.5)/128 과 점수 시그모이드는 HEF 안에 있다.
- 리사이즈: 비율 유지, 좌상단 기준, 우/하단 0 패딩. 원본 좌표 = 네트워크 좌표 / scale.
- 출력 9개: 스트라이드 8/16/32 (격자 80/40/20) x {점수 2ch, bbox 8ch, kps 20ch}, 셀당 앵커 2개
  (앵커 인덱스가 가장 빠르게 변한다). 레이어 이름은 모델마다 달라(scrfd_2.5g) (H, C) 로 고른다.
- 앵커 중심 (col*s, row*s), +0.5 오프셋 없음. bbox = 중심 -/+ 거리*s, kps = 중심 + 오프셋*s.
"""

import cv2
import numpy as np

INPUT_SIZE = 640
STRIDES = (8, 16, 32)
ANCHORS = 2
SCORE_THR = 0.5
NMS_IOU = 0.4


def _anchor_centers(stride):
    n = INPUT_SIZE // stride
    ys, xs = np.mgrid[:n, :n]
    centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
    return np.repeat(centers, ANCHORS, axis=0)


_CENTERS = {stride: _anchor_centers(stride) for stride in STRIDES}


def letterbox(bgr):
    """BGR 프레임 -> (RGB uint8 640x640, scale)."""
    h, w = bgr.shape[:2]
    longest = max(h, w)
    nw, nh = max(1, w * INPUT_SIZE // longest), max(1, h * INPUT_SIZE // longest)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    canvas[:nh, :nw] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return canvas, INPUT_SIZE / longest


def _heads(outputs):
    """{stride: (score (N,), bbox (N, 4), kps (N, 10))}."""
    by_shape = {(a.shape[0], a.shape[2]): a for a in outputs.values()}
    heads = {}
    for stride in STRIDES:
        n = INPUT_SIZE // stride
        heads[stride] = (by_shape[(n, ANCHORS)].reshape(-1),
                         by_shape[(n, ANCHORS * 4)].reshape(-1, 4),
                         by_shape[(n, ANCHORS * 10)].reshape(-1, 10))
    return heads


def _nms(boxes, scores, iou):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        inter = (np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
                 * np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None))
        overlap = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[overlap <= iou]
    return keep


def decode(outputs, scale, score_thr=SCORE_THR, iou=NMS_IOU):
    """원시 헤드 {이름: float32 (H, W, C)} -> 원본 좌표 [(bbox[4], score, kps[5, 2])], 면적이 큰 순."""
    boxes, scores, points = [], [], []
    for stride, (score, bbox, kps) in _heads(outputs).items():
        keep = score >= score_thr
        if not keep.any():
            continue
        centers = _CENTERS[stride][keep]
        dist = bbox[keep] * stride
        boxes.append(np.concatenate([centers - dist[:, :2], centers + dist[:, 2:]], axis=1))
        points.append(centers[:, None, :] + kps[keep].reshape(-1, 5, 2) * stride)
        scores.append(score[keep])
    if not boxes:
        return []
    boxes = np.concatenate(boxes) / scale
    points = np.concatenate(points) / scale
    scores = np.concatenate(scores)
    faces = [(boxes[i], float(scores[i]), points[i]) for i in _nms(boxes, scores, iou)]
    faces.sort(key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]), reverse=True)
    return faces
