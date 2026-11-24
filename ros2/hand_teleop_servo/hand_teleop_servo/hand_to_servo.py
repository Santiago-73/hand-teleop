"""Turns /hand/target and /hand/gripper_closed into MoveIt Servo pose commands and gripper goals."""
import rclpy
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PointStamped, PoseStamped
from moveit_msgs.srv import ServoCommandType
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool

# Gripper pointing down. This is the orientation of panda_link8 in the Panda "ready" pose
# (180 deg around X, -45 deg around Z). With (1, 0, 0, 0) the wrist spins 45 deg at startup.
# TODO: orientation from the hand, now it's fixed pointing down
DOWN_XYZW = [0.92388, -0.38268, 0.0, 0.0]


class HandToServo(Node):
    def __init__(self):
        super().__init__("hand_to_servo")
        self.declare_parameter("planning_frame", "panda_link0")
        self.declare_parameter("servo_node", "/servo_node")
        self.declare_parameter("orientation_xyzw", DOWN_XYZW)
        self.declare_parameter("gripper_action", "/panda_hand_controller/gripper_cmd")
        self.declare_parameter("gripper_open", 0.04)  # m per finger
        self.declare_parameter("gripper_closed", 0.0)
        self.declare_parameter("gripper_effort", 20.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.frame = g("planning_frame")
        self.q = [float(v) for v in g("orientation_xyzw")]
        servo = g("servo_node").rstrip("/")

        self.pub_pose = self.create_publisher(PoseStamped, f"{servo}/pose_target_cmds", 10)
        self.create_subscription(PointStamped, "hand/target", self.on_target, 10)
        self.create_subscription(Bool, "hand/gripper_closed", self.on_gripper, 10)

        # Servo starts in joint jog mode, switch to POSE once it's up
        self.switch_cli = self.create_client(ServoCommandType, f"{servo}/switch_command_type")
        self.servo_ready = False
        self.create_timer(1.0, self.ensure_pose_mode)

        self.gripper = ActionClient(self, GripperCommand, g("gripper_action"))
        self.gripper_state = None

    def ensure_pose_mode(self) -> None:
        if self.servo_ready:
            return
        if not self.switch_cli.service_is_ready():
            self.get_logger().info("Waiting for servo_node...", throttle_duration_sec=5.0)
            return
        req = ServoCommandType.Request(command_type=ServoCommandType.Request.POSE)
        self.switch_cli.call_async(req).add_done_callback(self._on_switched)
        self.servo_ready = True

    def _on_switched(self, fut) -> None:
        if fut.result() is not None and fut.result().success:
            self.get_logger().info("Servo in POSE mode")
        else:
            self.get_logger().warn("Could not switch Servo to POSE, retrying")
            self.servo_ready = False

    def on_target(self, msg: PointStamped) -> None:
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.frame
        pose.pose.position = msg.point
        o = pose.pose.orientation
        o.x, o.y, o.z, o.w = self.q
        self.pub_pose.publish(pose)

    def on_gripper(self, msg: Bool) -> None:
        # only send a goal when the state changes, otherwise it's a new goal every frame
        closed = bool(msg.data)
        if closed == self.gripper_state:
            return
        if not self.gripper.server_is_ready():
            self.get_logger().warn("Gripper action server not available", throttle_duration_sec=5.0)
            return
        self.gripper_state = closed
        goal = GripperCommand.Goal()
        goal.command.position = float(
            self.get_parameter("gripper_closed" if closed else "gripper_open").value)
        goal.command.max_effort = float(self.get_parameter("gripper_effort").value)
        self.gripper.send_goal_async(goal)
        self.get_logger().info("Gripper " + ("closing" if closed else "opening"))


def main():
    rclpy.init()
    node = HandToServo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
