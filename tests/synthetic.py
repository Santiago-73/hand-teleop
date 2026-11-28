"""Fake but anatomically plausible landmarks for tests (right hand, palm to the camera,
image coordinates, y down)."""
import numpy as np

KNUCKLES = {"thumb": (-0.06, -0.03), "index": (-0.03, -0.10), "middle": (0.0, -0.11),
            "ring": (0.03, -0.10), "pinky": (0.055, -0.08)}


def _finger(base, direction, lengths):
    pts, p = [], np.array(base, dtype=float)
    for length, d in zip(lengths, direction):
        p = p + length * np.array(d)
        pts.append(p.copy())
    return pts


def hand(pose: str, wrist=(0.5, 0.7), scale=1.0, rng=None) -> np.ndarray:
    w = np.array(wrist)
    out = [w]
    up, down = (0.0, -1.0), (0.0, 1.0)
    for name in ("thumb", "index", "middle", "ring", "pinky"):
        k = np.array(KNUCKLES[name]) * scale
        if name == "thumb":
            if pose == "thumbs_up":
                segs = [(-0.6, -0.8), up, up, up]
            elif pose in ("fist", "pinch"):
                segs = [(-0.6, -0.8), (0.7, -0.7), (1, 0), (1, 0)]
            else:
                segs = [(-0.6, -0.8), (-0.6, -0.8), (-0.6, -0.8), (-0.6, -0.8)]
            pts = _finger(w, segs, [0.035 * scale, 0.035 * scale, 0.03 * scale, 0.025 * scale])
            pts[0] = w + k
        else:
            curled = pose in ("fist", "thumbs_up") or (pose == "pinch" and name == "index")
            segs = [up, up, down, down] if curled else [up, up, up, up]
            pts = _finger(w, segs, [0.0, 0.04 * scale, 0.025 * scale, 0.02 * scale])
            pts = [w + k] + [w + k + (p - w) for p in pts[1:]]
        out += pts
    p = np.array(out)
    if pose == "thumbs_up":
        # a real thumbs up is a fist turned sideways with the thumb pointing up
        c, s_ = np.cos(np.pi / 2), np.sin(np.pi / 2)
        p = (p - w) @ np.array([[c, s_], [-s_, c]]) + w
        p[2:5] = p[1] + np.outer([1, 2, 3], [0.0, -0.035 * scale])
    if pose == "pinch":
        p[4] = p[8] + np.array([-0.005, 0.0]) * scale  # thumb tip touching index tip
    p = np.hstack([p, np.zeros((21, 1))]).astype(np.float32)
    if rng is not None:
        p[:, :2] += rng.normal(0, 0.002 * scale, (21, 2))
    return p
