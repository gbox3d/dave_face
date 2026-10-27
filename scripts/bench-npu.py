"""NPU 점검·벤치. dave-face 서비스(또는 다른 프로세스)가 NPU 를 잡고 있지 않을 때 실행한다.

  cd /home/goback/works/dave_face
  HAILORT_LOGGER_PATH=localfiles/logs apps/api/.venv/bin/python scripts/bench-npu.py          # 출력 이름·shape·양자화, 모델별 지연시간
  HAILORT_LOGGER_PATH=localfiles/logs apps/api/.venv/bin/python scripts/bench-npu.py --hold 60 # 60초 동안 VDevice 를 잡고 1초에 1회 추론
  ... --image a.jpg [--draw out.jpg]   # 검출 결과(점수·박스·5점·자세·블러)와 파이프라인 판정
  ... --pair a.jpg b.jpg [--bgr]        # 두 사진의 가장 큰 얼굴 임베딩 코사인 (--bgr: ArcFace 에 BGR 을 넣어 채널 순서 확인)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))
from dave_face import scrfd  # noqa: E402
from dave_face.npu import FaceNPU  # noqa: E402
from dave_face.pipeline import FacePipeline, align, pose  # noqa: E402


def timed(model, runs):
    image = np.zeros(model.input_shape, dtype=np.uint8)
    for _ in range(2):
        model.infer(image)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model.infer(image)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2], times[round(0.95 * (len(times) - 1))]


def hold(npu, seconds):
    det_in = np.zeros(npu.det.input_shape, dtype=np.uint8)
    rec_in = np.zeros(npu.rec.input_shape, dtype=np.uint8)
    done = errors = 0
    print(f"holding {seconds:g} s")
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            npu.detect_raw(det_in)
            npu.embed(rec_in)
            done += 1
        except Exception as error:  # 점검 도구: 실패를 세고 계속 돈다
            errors += 1
            print(f"error {type(error).__name__}: {error}")
        time.sleep(1)
    print(f"hold {seconds:g} s: {done} ok, {errors} errors")
    return errors


def read(path):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        sys.exit(f"cannot read image: {path}")
    return frame


def show_image(npu, path, draw):
    frame = read(path)
    canvas, scale = scrfd.letterbox(frame)
    start = time.perf_counter()
    faces = scrfd.decode(npu.detect_raw(canvas), scale)
    print(f"{path}: {frame.shape[1]}x{frame.shape[0]}, {len(faces)} face(s), detect+decode "
          f"{(time.perf_counter() - start) * 1000:.1f} ms")
    for bbox, score, kps in faces:
        eye_dist, roll, yaw = pose(kps)
        order_ok = kps[0, 0] < kps[1, 0] and max(kps[0, 1], kps[1, 1]) < kps[2, 1] < min(kps[3, 1], kps[4, 1])
        print(f"  score {score:.3f} bbox {np.round(bbox).astype(int).tolist()} eye_dist {eye_dist:.1f} "
              f"roll {roll:.1f} yaw {yaw:+.3f} kps_order {'ok' if order_ok else 'BAD'}")
        print(f"  kps {np.round(kps, 1).tolist()}")
    result = FacePipeline(npu).analyze(frame)
    print(f"  pipeline: {type(result).__name__} "
          + (result.reason if hasattr(result, "reason") else f"blur {result.blur:.0f}"))
    if draw:
        for bbox, _, kps in faces:
            x1, y1, x2, y2 = np.round(bbox).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x, y in np.round(kps).astype(int):
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        cv2.imwrite(str(draw), frame)
        if faces:
            cv2.imwrite(str(Path(draw).with_suffix(".crop.png")), cv2.cvtColor(align(read(path), faces[0][2]),
                                                                              cv2.COLOR_RGB2BGR))
        print(f"  drawn -> {draw}")


def embedding(npu, path, bgr):
    frame = read(path)
    canvas, scale = scrfd.letterbox(frame)
    faces = scrfd.decode(npu.detect_raw(canvas), scale)
    if not faces:
        sys.exit(f"no face: {path}")
    crop = align(frame, faces[0][2])
    return npu.embed(crop[..., ::-1] if bgr else crop)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--hold", type=float, default=0, help="VDevice 를 잡고 있을 초 (1초에 1회 추론)")
    parser.add_argument("--image", nargs="+", help="검출을 확인할 사진")
    parser.add_argument("--draw", help="--image 가 하나일 때 결과를 그려 저장할 경로")
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"), help="임베딩 코사인을 볼 두 사진")
    parser.add_argument("--bgr", action="store_true", help="--pair 에서 ArcFace 에 BGR 크롭을 넣는다")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # 파일로 돌려도 줄마다 보이게 (--hold 점검용)

    start = time.perf_counter()
    with FaceNPU() as npu:
        print(f"open {(time.perf_counter() - start) * 1000:.0f} ms")
        if args.image or args.pair:
            for path in args.image or []:
                show_image(npu, path, args.draw if len(args.image) == 1 else None)
            if args.pair:
                a, b = (embedding(npu, path, args.bgr) for path in args.pair)
                print(f"cos {float(a @ b):.4f} ({'BGR' if args.bgr else 'RGB'}) {args.pair[0]} vs {args.pair[1]}")
            return 0
        for label, model in (("det", npu.det), ("rec", npu.rec)):
            print(f"[{label}] input {model.input_shape}")
            for name, (shape, zp, scale) in model.outputs.items():
                print(f"[{label}] output {name} {shape} zp={zp:g} scale={scale:.6f}")
            median, p95 = timed(model, args.runs)
            print(f"[{label}] {args.runs} runs: median {median:.2f} ms, p95 {p95:.2f} ms")
        vector = npu.embed(np.zeros(npu.rec.input_shape, dtype=np.uint8))
        print(f"[rec] embed shape {vector.shape} dtype {vector.dtype} norm {np.linalg.norm(vector):.4f}")
        errors = hold(npu, args.hold) if args.hold else 0
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
