"""Train the gesture MLP on data/gestures.csv and export it to models/gesture_mlp.json.

    python train_gestures.py
    python train_gestures.py --hidden 64 32 --no-augment

Prints test accuracy and the confusion matrix, next to the rule-based classifier on
the same test frames, so it's clear whether the MLP is actually worth it.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from gestures import GESTURES, MlpClassifier, RuleClassifier, features

HERE = Path(__file__).parent


def load(path: Path):
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
    labels = raw[:, 0]
    is_left = raw[:, 1].astype(int).astype(bool)
    pts = raw[:, 2:].astype(np.float32).reshape(-1, 21, 3)
    return pts, is_left, labels


def augment(pts: np.ndarray, rng, copies: int = 3) -> np.ndarray:
    """Small in-plane rotations and squashes, like tilting the hand a bit in front of the camera."""
    out = [pts]
    for _ in range(copies):
        a = rng.uniform(-np.radians(15), np.radians(15), len(pts))
        sx = rng.uniform(0.9, 1.1, len(pts))
        c, s = np.cos(a), np.sin(a)
        p = pts - pts[:, :1]
        x, y = p[..., 0] * sx[:, None], p[..., 1]
        q = p.copy()
        q[..., 0] = c[:, None] * x - s[:, None] * y
        q[..., 1] = s[:, None] * x + c[:, None] * y
        out.append(q + rng.normal(0, 0.002, q.shape).astype(np.float32))
    return np.concatenate(out)


def export(clf: MLPClassifier, mean, std, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "classes": [str(c) for c in clf.classes_],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": [w.tolist() for w in clf.coefs_],
        "biases": [b.tolist() for b in clf.intercepts_],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=HERE / "data" / "gestures.csv")
    parser.add_argument("--out", type=Path, default=HERE / "models" / "gesture_mlp.json")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pts, is_left, labels = load(args.data)
    print(f"{len(labels)} samples: " + ", ".join(f"{g}={np.sum(labels == g)}" for g in GESTURES))

    idx_train, idx_test = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=args.seed)

    rng = np.random.default_rng(args.seed)
    tr_pts, tr_left, tr_y = pts[idx_train], is_left[idx_train], labels[idx_train]
    if not args.no_augment:
        n = len(tr_pts)
        tr_pts = augment(tr_pts, rng)
        reps = len(tr_pts) // n
        tr_left, tr_y = np.tile(tr_left, reps), np.tile(tr_y, reps)

    X_train = np.stack([features(p, l) for p, l in zip(tr_pts, tr_left)])
    X_test = np.stack([features(p, l) for p, l in zip(pts[idx_test], is_left[idx_test])])
    y_test = labels[idx_test]

    mean, std = X_train.mean(0), X_train.std(0) + 1e-6
    clf = MLPClassifier(hidden_layer_sizes=tuple(args.hidden), max_iter=500,
                        early_stopping=True, random_state=args.seed)
    clf.fit((X_train - mean) / std, tr_y)
    export(clf, mean, std, args.out)

    # evaluate the exported model, not the sklearn one, so the numpy inference is tested too
    mlp = MlpClassifier(args.out)
    present = [g for g in GESTURES if g in set(labels)]
    pred_mlp = np.array([mlp.classes[i] for i in mlp.proba(X_test).argmax(1)])
    rules = RuleClassifier()
    pred_rules = np.array([rules.predict(p, l)[0] for p, l in zip(pts[idx_test], is_left[idx_test])])

    print(f"\nTest frames: {len(y_test)}")
    print(f"MLP   accuracy: {np.mean(pred_mlp == y_test):.3f}")
    print(f"Rules accuracy: {np.mean(pred_rules == y_test):.3f}")
    print("\nMLP report:\n" + classification_report(y_test, pred_mlp, labels=present, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted): " + " ".join(present))
    print(confusion_matrix(y_test, pred_mlp, labels=present))

    t = time.perf_counter()
    for x in X_test[:500]:
        mlp.proba(x)
    per = (time.perf_counter() - t) / min(len(X_test), 500) * 1e6
    n_params = sum(w.size + b.size for w, b in mlp.layers)
    print(f"\n{n_params} parameters, {per:.0f} us per frame (numpy, one sample)")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
