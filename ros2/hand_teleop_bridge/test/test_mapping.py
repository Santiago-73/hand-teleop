import pytest

from hand_teleop_bridge.mapping import Clutch, Ema, Workspace, image_to_robot

WS = Workspace()


def test_center_of_image_is_center_of_workspace():
    x, y, z = image_to_robot(0.5, 0.5, (WS.size_far + WS.size_near) / 2, WS)
    assert abs(x - 0.40) < 1e-9 and abs(y) < 1e-9 and abs(z - 0.30) < 1e-9


def test_directions():
    # image right -> +Y, image top -> +Z, big hand -> +X
    assert image_to_robot(0.9, 0.5, 0.15, WS)[1] > 0
    assert image_to_robot(0.5, 0.1, 0.15, WS)[2] > 0.4
    assert image_to_robot(0.5, 0.5, 0.30, WS)[0] == WS.x_max


def test_clamped_to_workspace():
    x, y, z = image_to_robot(-1.0, 2.0, 0.0, WS)
    assert (x, y, z) == (WS.x_min, WS.y_min, WS.z_min)


def test_ema_smooths():
    f = Ema(0.5)
    assert f((0.0,)) == (0.0,)
    assert f((1.0,)) == (0.5,)


def test_clutch_passes_through_by_default():
    c = Clutch(WS)
    assert c((0.4, 0.0, 0.3), engaged=False) == (0.4, 0.0, 0.3)


def test_clutch_holds_and_continues_without_jump():
    c = Clutch(WS)
    c((0.4, 0.0, 0.3), engaged=False)
    # hand moves 10 cm left while the clutch is engaged: arm doesn't move
    assert c((0.4, 0.1, 0.3), engaged=True) == (0.4, 0.0, 0.3)
    # released: same spot, then follows the hand relative to there
    assert c((0.4, 0.1, 0.3), engaged=False) == pytest.approx((0.4, 0.0, 0.3))
    assert c((0.4, 0.15, 0.3), engaged=False) == pytest.approx((0.4, 0.05, 0.3))


def test_clutch_recenter_and_clamp():
    c = Clutch(WS)
    c((0.4, 0.0, 0.3), engaged=False)
    c.engage()  # hand lost
    c((0.4, 0.2, 0.3), engaged=False)
    c.recenter()
    assert c((0.4, 0.2, 0.3), engaged=False) == pytest.approx((0.4, 0.2, 0.3))
    assert c((9.0, -9.0, 9.0), engaged=False) == (WS.x_max, WS.y_min, WS.z_max)


def test_clutch_before_first_target_returns_none():
    assert Clutch(WS)((0.4, 0.0, 0.3), engaged=True) is None
