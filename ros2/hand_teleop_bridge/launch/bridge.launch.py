from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rviz = LaunchConfiguration("rviz")
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("port", default_value="5005"),
        Node(
            package="hand_teleop_bridge",
            executable="udp_bridge",
            output="screen",
            parameters=[{"port": LaunchConfiguration("port")}],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-f", "world"],
            condition=IfCondition(rviz),
        ),
    ])
