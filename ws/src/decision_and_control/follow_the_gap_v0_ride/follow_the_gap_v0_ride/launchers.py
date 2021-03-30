# NOTE: this is workaround for misbehaviour of symlink install
#       see the project README.md (the top-level one)
#       for more info (section [Launch files are not symlinked])
import launch
import launch.actions
import launch.substitutions
import launch_ros.actions


def generate_start_launch_description():
    return launch.LaunchDescription([

        launch_ros.actions.Node(
            package='follow_the_gap_v0',
            executable='follow_the_gap',
            output='screen',
            name=[launch.substitutions.AnonName('follow_the_gap')],
        ),

        launch_ros.actions.Node(
            package='follow_the_gap_v0_ride',
            executable='ride_node',
            output='screen',
            name=[launch.substitutions.AnonName('follow_the_gap_ride')],
        ),

    ])
