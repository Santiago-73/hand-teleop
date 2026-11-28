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
    # Image is already mirrored. Hand right -> +Y, hand up -> +Z, closer -> +X,
    # so it moves like a mirror when you face the robot (or the RViz view).
    rx = lerp_clamped(size, ws.size_far, ws.size_near, ws.x_min, ws.x_max)
    ry = lerp_clamped(x, 0.0, 1.0, ws.y_min, ws.y_max)
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


def clamp_to_workspace(p: tuple[float, float, float], ws: Workspace) -> tuple[float, float, float]:
    return (
        min(max(p[0], ws.x_min), ws.x_max),
        min(max(p[1], ws.y_min), ws.y_max),
        min(max(p[2], ws.z_min), ws.z_max),
    )


class Clutch:
    """Like the clutch pedal on a surgical robot. While engaged the arm holds still and the
    hand can move freely; on release the offset between hand and arm is updated so the arm
    continues from where it stopped instead of jumping to the hand.

    Losing the hand counts as engaging, so when it comes back the arm doesn't jump either.
    recenter() drops the offset: the arm goes back to the absolute hand position
    (Servo's speed limit keeps that move slow).
    """

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.offset = (0.0, 0.0, 0.0)
        self.engaged = False
        self.output = None

    def engage(self) -> None:
        self.engaged = True

    def recenter(self) -> None:
        self.offset = (0.0, 0.0, 0.0)

    def __call__(self, hand: tuple[float, float, float], engaged: bool):
        if engaged:
            self.engaged = True
            return self.output
        if self.engaged and self.output is not None:
            self.offset = tuple(o - h for o, h in zip(self.output, hand))
        self.engaged = False
        self.output = clamp_to_workspace(tuple(h + o for h, o in zip(hand, self.offset)), self.ws)
        return self.output
