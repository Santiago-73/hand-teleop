"""What each gesture does to the robot. Kept apart from the ROS node so it's testable
and so the real arm (ESP32) can reuse the same rules later.

  open       move, gripper open
  pinch      move, gripper closed
  fist       clutch: arm holds still, gripper keeps its state
  thumbs_up  recenter: drop the clutch offset
  other      move, gripper keeps its state
"""
from dataclasses import dataclass


@dataclass
class Command:
    clutch: bool
    gripper_closed: bool
    recenter: bool


def gesture_from_msg(m: dict) -> str:
    # older tracker versions only send "closed"
    return m.get("gesture") or ("pinch" if m.get("closed") else "open")


class GestureActions:
    def __init__(self):
        self.gripper_closed = False
        self.gesture = None

    def __call__(self, gesture: str) -> Command:
        if gesture == "pinch":
            self.gripper_closed = True
        elif gesture == "open":
            self.gripper_closed = False
        recenter = gesture == "thumbs_up" and self.gesture != "thumbs_up"
        self.gesture = gesture
        return Command(clutch=gesture == "fist", gripper_closed=self.gripper_closed, recenter=recenter)
