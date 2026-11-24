from glob import glob

from setuptools import find_packages, setup

package_name = "hand_teleop_servo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Santiago",
    description="Hand teleoperation of a simulated Panda with MoveIt Servo",
    license="MIT",
    entry_points={
        "console_scripts": [
            "hand_to_servo = hand_teleop_servo.hand_to_servo:main",
        ],
    },
)
