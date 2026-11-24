from hand_teleop_bridge.mapping import Ema, Workspace, image_to_robot

WS = Workspace()


def test_center_of_image_is_center_of_workspace():
    x, y, z = image_to_robot(0.5, 0.5, (WS.size_far + WS.size_near) / 2, WS)
    assert abs(x - 0.40) < 1e-9 and abs(y) < 1e-9 and abs(z - 0.30) < 1e-9


def test_directions():
    # image right -> -Y, image top -> +Z, big hand -> +X
    assert image_to_robot(0.9, 0.5, 0.15, WS)[1] < 0
    assert image_to_robot(0.5, 0.1, 0.15, WS)[2] > 0.4
    assert image_to_robot(0.5, 0.5, 0.30, WS)[0] == WS.x_max


def test_clamped_to_workspace():
    x, y, z = image_to_robot(-1.0, 2.0, 0.0, WS)
    assert (x, y, z) == (WS.x_min, WS.y_max, WS.z_min)


def test_ema_smooths():
    f = Ema(0.5)
    assert f((0.0,)) == (0.0,)
    assert f((1.0,)) == (0.5,)
