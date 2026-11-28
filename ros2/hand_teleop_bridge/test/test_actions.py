from hand_teleop_bridge.actions import GestureActions, gesture_from_msg


def test_gripper_follows_pinch_and_open():
    a = GestureActions()
    assert not a("open").gripper_closed
    assert a("pinch").gripper_closed
    assert not a("open").gripper_closed


def test_fist_clutches_and_keeps_the_gripper_closed():
    a = GestureActions()
    a("pinch")
    cmd = a("fist")
    assert cmd.clutch and cmd.gripper_closed
    assert a("other").gripper_closed


def test_recenter_only_once_per_thumbs_up():
    a = GestureActions()
    assert a("thumbs_up").recenter
    assert not a("thumbs_up").recenter
    a("open")
    assert a("thumbs_up").recenter


def test_old_tracker_messages():
    assert gesture_from_msg({"closed": True}) == "pinch"
    assert gesture_from_msg({"closed": False}) == "open"
    assert gesture_from_msg({"closed": True, "gesture": "fist"}) == "fist"
