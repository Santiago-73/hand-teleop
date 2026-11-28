"""Static hand gesture classification from the 21 MediaPipe landmarks.

Two classifiers with the same interface:
  - RuleClassifier: finger extended/folded heuristics, works without any training
  - MlpClassifier: small MLP trained on my own recordings (train_gestures.py)

Inference is plain numpy on purpose, the weights are a JSON file so the same model
can be ported to C later without dragging scikit-learn along.
"""
import json
from pathlib import Path

import numpy as np

GESTURES = ["other", "open", "pinch", "fist", "thumbs_up"]

WRIST, THUMB_IP, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP = 0, 3, 4, 5, 8, 9
FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)
PINCH_THRESHOLD = 0.35


def to_array(landmarks) -> np.ndarray:
    """MediaPipe landmark list -> (21, 3) array."""
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)


def normalize(pts: np.ndarray, is_left: bool = False) -> np.ndarray:
    """Wrist at the origin, scaled by wrist->middle knuckle. Left hands mirrored so both
    look like a right hand. Position in the image and distance to the camera drop out."""
    p = pts - pts[WRIST]
    scale = np.linalg.norm(p[MIDDLE_MCP]) or 1e-6
    p = p / scale
    if is_left:
        p[:, 0] = -p[:, 0]
    return p


def features(pts: np.ndarray, is_left: bool = False) -> np.ndarray:
    return normalize(pts, is_left).reshape(-1)


def pinch_ratio(pts: np.ndarray) -> float:
    hand = np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]) or 1e-6
    return float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / hand)


class RuleClassifier:
    name = "rules"

    def predict(self, pts: np.ndarray, is_left: bool = False) -> tuple[str, float]:
        p = normalize(pts, is_left)
        d = np.linalg.norm(p, axis=1)  # distance of every point to the wrist
        extended = [d[t] > d[j] * 1.1 for t, j in zip(FINGER_TIPS, FINGER_PIPS)]

        if not any(extended):
            # thumbs up = thumb straight and clearly the highest point (image y grows downwards)
            thumb_straight = d[THUMB_TIP] > d[THUMB_IP]
            others_top = np.delete(p[:, 1], [2, THUMB_IP, THUMB_TIP]).min()
            thumb_up = p[THUMB_TIP, 1] < others_top - 0.25
            return ("thumbs_up", 1.0) if thumb_straight and thumb_up else ("fist", 1.0)
        if pinch_ratio(pts) < PINCH_THRESHOLD:
            return "pinch", 1.0
        if all(extended):
            return "open", 1.0
        return "other", 1.0


class MlpClassifier:
    name = "mlp"

    def __init__(self, path: Path):
        m = json.loads(Path(path).read_text())
        self.classes = m["classes"]
        self.mean = np.array(m["mean"], dtype=np.float32)
        self.std = np.array(m["std"], dtype=np.float32)
        self.layers = [(np.array(w, dtype=np.float32), np.array(b, dtype=np.float32))
                       for w, b in zip(m["weights"], m["biases"])]

    def proba(self, x: np.ndarray) -> np.ndarray:
        h = (x - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        e = np.exp(h - h.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def predict(self, pts: np.ndarray, is_left: bool = False) -> tuple[str, float]:
        p = self.proba(features(pts, is_left))
        i = int(np.argmax(p))
        return self.classes[i], float(p[i])


def load_classifier(model_path: Path | None, force_rules: bool = False):
    if not force_rules and model_path and Path(model_path).exists():
        return MlpClassifier(model_path)
    return RuleClassifier()


class Debouncer:
    """Only switches to a new gesture after it has been seen `frames` times in a row.
    Low-confidence predictions count as "other". One wrong frame would otherwise
    open the gripper in the middle of a grasp."""

    def __init__(self, frames: int = 4, min_conf: float = 0.7, initial: str = "other"):
        self.frames = frames
        self.min_conf = min_conf
        self.state = initial
        self._candidate = initial
        self._count = 0

    def reset(self) -> None:
        self.state = self._candidate = "other"
        self._count = 0

    def __call__(self, gesture: str, conf: float = 1.0) -> str:
        if conf < self.min_conf:
            gesture = "other"
        if gesture == self.state:
            self._count = 0
            return self.state
        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = gesture, 1
        if self._count >= self.frames:
            self.state, self._count = gesture, 0
        return self.state
