"""TODO"""

import os
from typing import List

from ament_index_python import get_package_share_directory
from launch import LaunchDescription, SomeSubstitutionsType
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def get_shared_file(package_name: str, *path: List[str]) -> str:
    """Return the path to the given shared file."""
    return os.path.join(
        get_package_share_directory(package_name),
        *path,
    )


def get_shared_file_substitution(package_name: str, *path: List[str]) -> SomeSubstitutionsType:
    """Return a Substitution that resolves to the path to the given shared file."""
    return PathJoinSubstitution([
        FindPackageShare(package=package_name),
        *path,
    ])


THIS_PKG = 'cartographer_slam'


def generate_launch_description():

    # https://github.com/SteveMacenski/slam_toolbox
    slam_toolbox_node = Node(
        executable='async_slam_toolbox_node',
        name='async_slam_toolbox_node',
        namespace='slam',
        package='slam_toolbox',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
            },
            get_shared_file_substitution(
                 'slam_toolbox_slam', 'config', 'slam.yaml'
            ),
        ],
        arguments=[

        ],
        remappings=[
            # see https://google-cartographer-ros.readthedocs.io/en/latest/ros_api.html
            # ('echoes', 'horizontal_laser_2d'),
            # ('scan', '/scan'),
            # ('odom', '/stage_ros2/car/odom'),
            # ('/imu', 'xxx'),
        ],
    )

    tf = Node(
        executable='static_transform_publisher',
        name='static_transform_publisher_imu',
        package='tf2_ros',
        output='screen',
        arguments=[
            # x y z qx qy qz qw
            '0', '0', '0', '0', '0', '0',
            # frame_id child_frame_id
            # 'car/odom', 'car/ranger_0/base_scan',
            # 'car/odom', 'odom',
            'odom', 'car/odom',
        ],
    )

    return LaunchDescription([
        slam_toolbox_node,
        # tf
    ])
