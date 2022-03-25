import os
from typing import List

from ament_index_python import get_package_share_directory
from launch import LaunchDescription, SomeSubstitutionsType
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration, LogInfo, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, ThisLaunchFileDir
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


def generate_launch_description():

    stage_launch = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            launch_file_path=PathJoinSubstitution([
                ThisLaunchFileDir(),
                'stage.launch.py',
            ]),
        ),
        launch_arguments=[
            ('world', LaunchConfiguration(variable_name='world')),
        ],
    )

    # TODO: refactor (declarative way, avoid side effects, use dynamic substition)
    urdf_path = get_shared_file('cartographer_slam', 'urdf', 'tx2-auto-3.urdf')
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
                'use_sim_time': True,
            },
        ],
    )

    # see https://docs.ros.org/en/rolling/Tutorials/Tf2/Writing-A-Tf2-Static-Broadcaster-Py.html#the-proper-way-to-publish-static-transforms
    tf_stage_base = Node(
        executable='static_transform_publisher',
        name='tf_stage_base',
        package='tf2_ros',
        output='screen',
        arguments=[
            # x y z qx qy qz qw
            '0', '0', '0', '0', '0', '0',
            # frame_id child_frame_id
            'car/base_link', 'base_link',
        ],
    )
    tf_stage_laser = Node(
        executable='static_transform_publisher',
        name='tf_stage_laser',
        package='tf2_ros',
        output='screen',
        arguments=[
            # x y z qx qy qz qw
            '0', '0', '0', '0', '0', '0',
            # frame_id child_frame_id
            'car/ranger_0/base_scan', 'laser',
        ],
    )

    obstacle_substitution_node = Node(
        package='obstacle_substitution',
        executable='obstacle_substitution_node',
        name='obstacle_substitution_node',
        remappings=[
            # ('/scan', ''),
            # ('/obstacles', ''),
        ],
        parameters=[
            {
                'use_sim_time': True,
            },
        ],
        output='screen',
    )

    follow_the_gap_v0_ride_launch = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            launch_file_path=get_shared_file_substitution(
                'follow_the_gap_v0_ride', 'start.launch.py',
            ),
        ),
        launch_arguments=[],
    )

    drive_api_node = Node(
        package='drive_api',
        executable='drive_api_node',
        name='drive_api_sim_node',
        parameters=[
            get_shared_file_substitution(
                'drive_api', 'config', 'drive_api_node.yaml',
            ),
            {
                'run_mode': 'simulation',
                'simulation.steering_modifier': 0.7,
                'simulation.throttle_modifier': 15.00,
            },
            {
                'use_sim_time': True,
            },
        ],
        remappings=[
            ('/cmd_vel', '/stage_ros2/car/cmd_vel'),
        ],
        output='screen',
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            name='world',
            description='World file name (incl. ext). Can be a relative path (with respect to storage/stage/world) or an absolute path.',
            default_value='test_track.world',
        ),

        stage_launch,
        # urdf_publisher,
        # tf_stage_base,
        # tf_stage_laser,
        # TODO: Is it needed with stage_ros2?
        # <!-- Transformation -->
        # <node
        # 	pkg="tf" type="static_transform_publisher"
        # 	name="tf_base_laser_link"
        # 	args="0 0 0 0 0 0 base_laser_link car/ranger_0/base_scan"
        # />
        # <node
        # 	pkg="tf"
        # 	type="static_transform_publisher"
        # 	name="tf_odom_map"
        # 	args="0 0 0 0 0 0 odom map"
        # />
        # TODO: rviz
        # TODO: map
        obstacle_substitution_node,
        follow_the_gap_v0_ride_launch,
        drive_api_node,
        # urdf_publisher,
        # tf_stage_laser,
    ])
