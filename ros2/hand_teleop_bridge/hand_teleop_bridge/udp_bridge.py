"""Receives the hand from hand_tracker.py over UDP and publishes it on /hand/*.

/hand/target, /hand/gripper_closed, /hand/pinch, /hand/gesture, /hand/clutch, /hand/marker
"""
import json
import socket

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker

from hand_teleop_bridge.actions import GestureActions, gesture_from_msg
from hand_teleop_bridge.mapping import Clutch, Ema, Workspace, image_to_robot


class UdpBridge(Node):
    def __init__(self):
        super().__init__("hand_udp_bridge")
        ws_defaults = Workspace()
        self.declare_parameter("port", 5005)
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("rate_hz", 60.0)
        self.declare_parameter("timeout", 0.5)  # stop publishing the target if no hand for this long
        self.declare_parameter("smoothing", 0.35)  # 1 = no filter, 0.1 = very smooth but laggy
        for name, value in vars(ws_defaults).items():
            self.declare_parameter(name, value)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.frame_id = g("frame_id")
        self.timeout = g("timeout")
        self.ws = Workspace(**{n: float(g(n)) for n in vars(ws_defaults)})
        self.ema = Ema(g("smoothing"))
        self.clutch = Clutch(self.ws)
        self.actions = GestureActions()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", g("port")))
        self.sock.setblocking(False)

        self.pub_target = self.create_publisher(PointStamped, "hand/target", 10)
        self.pub_closed = self.create_publisher(Bool, "hand/gripper_closed", 10)
        self.pub_pinch = self.create_publisher(Float32, "hand/pinch", 10)
        self.pub_marker = self.create_publisher(Marker, "hand/marker", 10)
        self.pub_gesture = self.create_publisher(String, "hand/gesture", 10)
        self.pub_clutch = self.create_publisher(Bool, "hand/clutch", 10)

        self.last_msg = None
        self.last_rx = None
        self.lost_logged = True
        self.create_timer(1.0 / g("rate_hz"), self.tick)
        self.get_logger().info(f"Listening for UDP on 0.0.0.0:{g('port')}")

    def drain_socket(self) -> None:
        # only the newest packet matters, drop the backlog
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            try:
                self.last_msg = json.loads(data)
                self.last_rx = self.get_clock().now()
            except json.JSONDecodeError:
                self.get_logger().warn("Bad UDP packet", throttle_duration_sec=2.0)

    def tick(self) -> None:
        self.drain_socket()
        m = self.last_msg
        fresh = (
            m is not None
            and m.get("detected")
            and (self.get_clock().now() - self.last_rx).nanoseconds * 1e-9 < self.timeout
        )
        if not fresh:
            if not self.lost_logged:
                self.get_logger().info("Hand lost")
                self.lost_logged = True
            self.ema.reset()
            self.clutch.engage()  # so the arm doesn't jump when the hand comes back
            return
        if self.lost_logged:
            self.get_logger().info("Hand detected")
            self.lost_logged = False

        gesture = gesture_from_msg(m)
        if gesture != self.actions.gesture:
            self.get_logger().info(f"Gesture: {gesture}")
        cmd = self.actions(gesture)
        if cmd.recenter:
            self.clutch.recenter()
            self.get_logger().info("Recentered")
        hand = self.ema(image_to_robot(m["x"], m["y"], m["size"], self.ws))
        out = self.clutch(hand, cmd.clutch)
        self.pub_gesture.publish(String(data=gesture))
        self.pub_clutch.publish(Bool(data=cmd.clutch))
        if out is None:
            return
        x, y, z = out
        closed = cmd.gripper_closed
        stamp = self.get_clock().now().to_msg()

        target = PointStamped()
        target.header.stamp = stamp
        target.header.frame_id = self.frame_id
        target.point.x, target.point.y, target.point.z = x, y, z
        self.pub_target.publish(target)
        self.pub_closed.publish(Bool(data=closed))
        self.pub_pinch.publish(Float32(data=float(m["pinch"])))

        marker = Marker()
        marker.header = target.header
        marker.ns, marker.id = "hand", 0
        marker.type, marker.action = Marker.SPHERE, Marker.ADD
        marker.pose.position = target.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05
        marker.color.a = 1.0
        # red = closed, green = open, yellow = clutch
        marker.color.r, marker.color.g = (1.0, 1.0) if cmd.clutch else (1.0, 0.0) if closed else (0.0, 1.0)
        self.pub_marker.publish(marker)


def main():
    rclpy.init()
    node = UdpBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
