from typing import List

from launch import LaunchDescription, SomeSubstitutionsType
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration, LogInfo
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def get_shared_file_substitution(package_name: str, *path: List[str]) -> SomeSubstitutionsType:
    """Return a Substitution that resolves to the path to the given shared file."""
    return PathJoinSubstitution([
        FindPackageShare(package=package_name),
        *path,
    ])


def generate_launch_description():

    simulator_node = Node(
        package='stage_ros2',
        executable='stage_ros2',
        # note: do not change name or namespace otherwise generated node name for the car will be invalid
        # by default it creates one node for the simulator and one for each mode (car):
        # /stage_ros2
        # /stage_ros2/car
        parameters=[
            {
                'world': LaunchConfiguration(variable_name='world'),
            },
        ],
        remappings=[
            ('/stage_ros2/car/ranger_0', '/scan'),
        ],
        output='screen',
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            name='world',
            description='World file name (incl. ext). Can be a relative path (with respect to storage/stage/world) or an absolute path.',
        ),

        SetLaunchConfiguration(
            name='world',
            value=get_shared_file_substitution(
                'storage', 'stage', 'world',
                LaunchConfiguration(variable_name='world')
            ),
        ),

        LogInfo(
            msg=[
                'path to the world file = ',
                LaunchConfiguration(variable_name='world')
            ],
        ),

        simulator_node,

    ])
