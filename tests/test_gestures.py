import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gestures import Debouncer, MlpClassifier, RuleClassifier, features
from tests.synthetic import hand

ROOT = Path(__file__).parents[1]
POSES = ["open", "pinch", "fist", "thumbs_up"]


@pytest.mark.parametrize("pose", POSES)
def test_rules_recognise_synthetic_poses(pose):
    assert RuleClassifier().predict(hand(pose))[0] == pose


def test_features_ignore_position_and_distance():
    a = features(hand("open", wrist=(0.3, 0.6), scale=0.7))
    b = features(hand("open", wrist=(0.7, 0.8), scale=1.4))
    assert np.allclose(a, b, atol=1e-5)


def test_left_hand_is_mirrored():
    right = hand("thumbs_up")
    left = right.copy()
    left[:, 0] = 1.0 - left[:, 0]
    assert np.allclose(features(right), features(left, is_left=True), atol=1e-5)


def test_debouncer_needs_consecutive_frames():
    d = Debouncer(frames=3)
    assert [d("pinch") for _ in range(3)] == ["other", "other", "pinch"]
    assert d("open") == "pinch"           # one stray frame doesn't switch
    assert d("pinch") == "pinch"
    assert d("fist", conf=0.3) == "pinch"  # low confidence counts as "other"


def test_train_and_load_mlp(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "gestures.csv"
    with data.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "is_left"] + [f"{a}{i}" for i in range(21) for a in "xyz"])
        for pose in POSES:
            for _ in range(60):
                pts = hand(pose, wrist=rng.uniform(0.3, 0.7, 2), scale=rng.uniform(0.7, 1.3), rng=rng)
                w.writerow([pose, 0] + list(pts.reshape(-1)))
    model = tmp_path / "mlp.json"
    out = subprocess.run([sys.executable, str(ROOT / "train_gestures.py"), "--data", str(data), "--out", str(model)],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "MLP   accuracy: 1.000" in out.stdout
    mlp = MlpClassifier(model)
    assert mlp.predict(hand("fist", rng=rng))[0] == "fist"
