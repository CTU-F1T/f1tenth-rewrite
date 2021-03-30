# NOTE: this is workaround for misbehaviour of symlink install
#       see the project README.md (the top-level one)
#       for more info (section [Launch files are not symlinked])
import launch
import launch.actions
import launch.substitutions
import launch_ros.actions


def generate_stage_launch_description():
    return launch.LaunchDescription([

        launch.actions.DeclareLaunchArgument(
            name='world',
            description='Name of a world within storage/stage/world',
        ),

        launch.actions.SetLaunchConfiguration(
            name='world',
            value=[
                launch_ros.substitutions.FindPackageShare(package='storage'),
                '/stage/world/',
                launch.substitutions.LaunchConfiguration(variable_name='world'),
                '.world',
            ],
        ),

        launch.actions.LogInfo(
            msg=['path to world = ', launch.substitutions.LaunchConfiguration(variable_name='world')],
        ),

        launch_ros.actions.Node(
            package='stage_ros2',
            executable='stage_ros2',
            output='screen',
            name='stage_simulator',
            parameters=[
                {
                    'world': launch.substitutions.LaunchConfiguration(variable_name='world'),
                }
            ],
        ),

    ])
