# NOTE: this is workaround for misbehaviour of symlink install
#       see the project README.md (the top-level one)
#       for more info (section [Launch files are not symlinked])
import launch
import launch.actions
import launch.substitutions
import launch_ros.actions


# see https://docs.ros.org/en/foxy/Tutorials/Launch-system.html
# see https://github.com/ros2/launch/blob/foxy/launch/doc/source/architecture.rst
# see https://docs.ros.org/en/foxy/Guides/Node-arguments.html
def generate_start_launch_description():
    return launch.LaunchDescription({
        launch.actions.DeclareLaunchArgument(
            'node_prefix',
            default_value=[launch.substitutions.EnvironmentVariable('USER'), '_uiop_'],
            description='Prefix for node names',
        ),
        launch_ros.actions.Node(
            package='drive_api',
            executable='drive_api_node',
            output='screen',
            name=[launch.substitutions.LaunchConfiguration('node_prefix'), 'talker'],
            arguments={
                'use_simulation': True,
            },
        ),
    })
