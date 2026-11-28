"""Record labelled hand landmarks from the webcam for train_gestures.py.

Press a number to start recording that gesture, space to stop, q to quit:
  0 other   1 open   2 pinch   3 fist   4 thumbs_up
Move the hand around while recording (closer, further, rotated a bit), that's what
makes the model generalize. Rows get appended to data/gestures.csv.
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from gestures import GESTURES, to_array
from hand_tracker import CAMERA_INDEX, draw_hand, ensure_model

DATA = Path(__file__).with_name("data") / "gestures.csv"


def is_left_hand(result) -> bool:
    return result.handedness[0][0].category_name == "Left"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DATA)
    parser.add_argument("--every", type=int, default=2, help="keep one frame out of N")
    args = parser.parse_args()

    args.out.parent.mkdir(exist_ok=True)
    new_file = not args.out.exists() or args.out.stat().st_size == 0
    counts = Counter()
    if not new_file:
        with args.out.open() as f:
            counts.update(row[0] for row in csv.reader(f) if row and row[0] != "label")
    f = args.out.open("a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label", "is_left"] + [f"{a}{i}" for i in range(21) for a in "xyz"])

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording = None
    frame_i = 0
    t0 = time.monotonic()
    last_ts = -1
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int((time.monotonic() - t0) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            frame_i += 1
            if result.hand_landmarks:
                draw_hand(frame, result.hand_landmarks[0], False)
                if recording is not None and frame_i % args.every == 0:
                    pts = to_array(result.hand_landmarks[0])
                    writer.writerow([recording, int(is_left_hand(result))] + [f"{v:.5f}" for v in pts.reshape(-1)])
                    counts[recording] += 1

            status = f"REC {recording}" if recording else "paused (0-4 to record, space to stop)"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255) if recording else (255, 255, 255), 2)
            for i, g in enumerate(GESTURES):
                cv2.putText(frame, f"{i} {g}: {counts[g]}", (10, 70 + 28 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Collect gestures (q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = None
                f.flush()
            elif ord("0") <= key < ord("0") + len(GESTURES):
                recording = GESTURES[key - ord("0")]

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved to {args.out}: {dict(counts)}")


if __name__ == "__main__":
    main()
