"""Webcam hand tracking with MediaPipe: draws the 21 landmarks, pinch state and FPS. q to quit."""
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
CAMERA_INDEX = 0
PINCH_THRESHOLD = 0.35  # tuned by hand, works for me

# 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        print(f"Downloading model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def pinch_ratio(lm) -> float:
    # normalized by hand size so it doesn't change with distance to the camera
    hand_size = dist(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6
    return dist(lm[THUMB_TIP], lm[INDEX_TIP]) / hand_size


def draw_hand(frame, lm, closed: bool) -> None:
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for i, (x, y) in enumerate(pts):
        color = (0, 0, 255) if i in (THUMB_TIP, INDEX_TIP) else (0, 200, 0)
        cv2.circle(frame, (x, y), 5, color, -1)
    if closed:
        cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], (0, 0, 255), 3)


def main() -> None:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # DSHOW opens much faster on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam, try CAMERA_INDEX = 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    t0 = time.monotonic()
    prev = t0
    fps = 0.0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Dropped frame, retrying...")
                continue
            frame = cv2.flip(frame, 1)  # mirrored is way easier to control

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - t0) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                ratio = pinch_ratio(lm)
                closed = ratio < PINCH_THRESHOLD
                draw_hand(frame, lm, closed)
                wrist = lm[WRIST]
                cv2.putText(frame, f"Wrist x={wrist.x:.2f} y={wrist.y:.2f} z={wrist.z:.3f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Pinch {'CLOSED' if closed else 'open'} ({ratio:.2f})",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255) if closed else (0, 200, 0), 2)
            else:
                cv2.putText(frame, "No hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand tracker (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
