"""Image coordinates -> robot workspace. No ROS in here so I can test it with plain pytest."""
from dataclasses import dataclass


def lerp_clamped(value: float, in_min: float, in_max: float, out_a: float, out_b: float) -> float:
    t = (value - in_min) / (in_max - in_min)
    t = min(max(t, 0.0), 1.0)
    return out_a * (1.0 - t) + out_b * t


@dataclass
class Workspace:
    # box in meters, robot base frame
    x_min: float = 0.25   # forward/back <- hand size (depth)
    x_max: float = 0.55
    y_min: float = -0.30  # left/right <- image x
    y_max: float = 0.30
    z_min: float = 0.10   # up/down <- image y
    z_max: float = 0.50
    # apparent hand size, small = far from the camera
    size_far: float = 0.08
    size_near: float = 0.22


def image_to_robot(x: float, y: float, size: float, ws: Workspace) -> tuple[float, float, float]:
    # Image is already mirrored. Hand right -> -Y, hand up -> +Z, closer -> +X,
    # so it feels like standing behind the robot.
    rx = lerp_clamped(size, ws.size_far, ws.size_near, ws.x_min, ws.x_max)
    ry = lerp_clamped(x, 0.0, 1.0, ws.y_max, ws.y_min)
    rz = lerp_clamped(y, 0.0, 1.0, ws.z_max, ws.z_min)
    return rx, ry, rz


class Ema:
    """Exponential moving average. Higher alpha = faster but more jitter."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def reset(self) -> None:
        self.value = None

    def __call__(self, sample: tuple[float, ...]) -> tuple[float, ...]:
        if self.value is None:
            self.value = sample
        else:
            a = self.alpha
            self.value = tuple(a * s + (1 - a) * v for s, v in zip(sample, self.value))
        return self.value
