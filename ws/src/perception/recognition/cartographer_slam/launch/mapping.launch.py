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

    # https://github.com/ros2/cartographer_ros
    cartographer_node = Node(
        executable='cartographer_node',
        name='cartographer_node',
        namespace='cartographer',
        package='cartographer_ros',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
            },
        ],
        arguments=[
            '-configuration_directory',
            get_shared_file_substitution(THIS_PKG, 'config'),
            '-configuration_basename',
            'mapping.lua',
        ],
        remappings=[
            # see https://google-cartographer-ros.readthedocs.io/en/latest/ros_api.html
            # ('echoes', 'horizontal_laser_2d'),
            ('scan', '/scan'),
            ('odom', '/stage_ros2/car/odom'),
            # ('/imu', 'xxx'),
        ],
    )

    occupancy_grid_node = Node(
        executable='cartographer_occupancy_grid_node',
        name='occupancy_grid_node',
        namespace='cartographer',
        package='cartographer_ros',
        output='screen',
        arguments=[
            '-resolution',
            '0.05',
        ],
        parameters=[
            {
                'use_sim_time': True,
            },
        ],
    )

    # see https://docs.ros.org/en/rolling/Tutorials/Tf2/Writing-A-Tf2-Static-Broadcaster-Py.html#the-proper-way-to-publish-static-transforms
    static_transform_publisher_laser = Node(
        executable='static_transform_publisher',
        name='static_transform_publisher_laser',
        package='tf2_ros',
        output='screen',
        arguments=[
            # x y z qx qy qz qw
            '0', '0', '0', '0', '0', '0',
            # frame_id child_frame_id
            'base_link', 'laser',
        ],
    )

    # see https://docs.ros.org/en/rolling/Tutorials/Tf2/Writing-A-Tf2-Static-Broadcaster-Py.html#the-proper-way-to-publish-static-transforms
    static_transform_publisher_imu = Node(
        executable='static_transform_publisher',
        name='static_transform_publisher_imu',
        package='tf2_ros',
        output='screen',
        arguments=[
            # x y z qx qy qz qw
            '0', '0', '0', '0', '-0.707', '0.707',
            # frame_id child_frame_id
            'base_link', 'base_imu_link',
        ],
    )

    # TODO: refactor (declarative way, avoid side effects, use dynamic substition)
    urdf_path = get_shared_file(THIS_PKG, 'urdf', 'tx2-auto-3.urdf')
    with open(urdf_path, 'r') as urdf_file_handle:
        urdf_file_content = urdf_file_handle.read()

    urdf_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                # Parameter robot_description as file path, not URDF string
                # https://github.com/ros/robot_state_publisher/issues/155
                'robot_description': urdf_file_content,
            },
        ],
    )

    return LaunchDescription([
        cartographer_node,
        occupancy_grid_node,
        # urdf_publisher,
        # static_transform_publisher_laser,
        # static_transform_publisher_imu,
    ])
