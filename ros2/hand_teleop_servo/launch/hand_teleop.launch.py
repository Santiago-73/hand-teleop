# Simulated Panda (ros2_control mock hardware) + MoveIt Servo + UDP hand bridge + RViz.
# Started from moveit_servo/launch/demo_ros_api.launch.py
#
#   ros2 launch hand_teleop_servo hand_teleop.launch.py
#   ros2 launch hand_teleop_servo hand_teleop.launch.py rviz:=false port:=5005
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_param_builder import ParameterBuilder
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

# Box the hand gets mapped to (m, panda_link0). With the gripper pointing down the Panda
# reaches all of it without getting close to its own base or the floor.
WORKSPACE = {
    "x_min": 0.30, "x_max": 0.60,
    "y_min": -0.30, "y_max": 0.30,
    "z_min": 0.20, "z_max": 0.60,
}


def generate_launch_description():
    rviz = LaunchConfiguration("rviz")
    port = LaunchConfiguration("port")

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(file_path="config/panda.urdf.xacro")
        .joint_limits(file_path="config/hard_joint_limits.yaml")
        .to_moveit_configs()
    )
    servo_params = {
        "moveit_servo": ParameterBuilder("hand_teleop_servo").yaml("config/servo.yaml").to_dict()
    }
    ros2_controllers = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "config", "ros2_controllers.yaml",
    )
    rviz_config = os.path.join(
        get_package_share_directory("hand_teleop_servo"), "config", "hand_teleop.rviz")

    def spawner(name):
        return Node(
            package="controller_manager", executable="spawner",
            arguments=[name, "-c", "/controller_manager", "--controller-manager-timeout", "60"],
        )

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("port", default_value="5005"),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[moveit_config.robot_description]),
        Node(package="tf2_ros", executable="static_transform_publisher",
             arguments=["--frame-id", "world", "--child-frame-id", "panda_link0"]),
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[ros2_controllers],
             remappings=[("/controller_manager/robot_description", "/robot_description")],
             output="screen"),
        spawner("joint_state_broadcaster"),
        spawner("panda_arm_controller"),
        spawner("panda_hand_controller"),

        Node(
            package="moveit_servo", executable="servo_node", name="servo_node",
            parameters=[
                servo_params,
                {"update_period": 0.01, "planning_group_name": "panda_arm"},
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.joint_limits,
            ],
            output="screen",
        ),

        # target goes out in panda_link0 directly, Servo can't use "world" (it's not in the robot model)
        Node(package="hand_teleop_bridge", executable="udp_bridge", output="screen",
             parameters=[{"port": port, "frame_id": "panda_link0", **WORKSPACE}]),
        Node(package="hand_teleop_servo", executable="hand_to_servo", output="screen"),

        Node(package="rviz2", executable="rviz2", output="log",
             arguments=["-d", rviz_config],
             parameters=[moveit_config.robot_description,
                         moveit_config.robot_description_semantic],
             condition=IfCondition(rviz)),
    ])
