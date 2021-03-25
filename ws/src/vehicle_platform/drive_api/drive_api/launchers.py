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
    defaults = {
        'anonymous': 'false',
        'simulation': 'false',
        'use_vesc': 'false',
        'rate': '10'
    }

    return launch.LaunchDescription({

        # TODO: Is this a bug?
        #   The following does not work
        #     `SubstitutionFailure: launch configuration 'anonymous' does not exist` is thrown
        #     when attempting to use it like launch.substitutions.LaunchConfiguration('anonymous')
        #     launch.actions.DeclareLaunchArgument(
        #         'anonymous',
        #         default_value='false',
        #         description='When true, run the node as anonymous (generate random name).',
        #     ),

        launch.actions.DeclareLaunchArgument(
            'anonymous',
            default_value=defaults['anonymous'],
            description='When true, run the node as anonymous (generate random name).',
        ),
        launch.actions.DeclareLaunchArgument(
            'simulation',
            default_value=defaults['simulation'],
            description=(
                'When true, run the node in simulation mode. '
                'If anonymous=true then it the simulation=true is forced no matter what.'
            ),
        ),
        launch.actions.DeclareLaunchArgument(
            'use_vesc',
            default_value=defaults['use_vesc'],
            # TODO: Maybe change "rather than" to "instead of"?
            description='When true speed is published directly to VESC rather than to Teensy.',
        ),
        launch.actions.DeclareLaunchArgument(
            'rate',
            default_value=defaults['rate'],
            description='Publish rate of the Drive-API messages.',
        ),

        launch.actions.SetLaunchConfiguration(
            'node_name',
            value=launch.substitutions.AnonName(name='drive_api'),
            condition=launch.conditions.IfCondition(
                launch.substitutions.LaunchConfiguration('anonymous', default=defaults['anonymous'])
            ),
        ),
        launch.actions.SetLaunchConfiguration(
            'node_name',
            value='drive_api',
            condition=launch.conditions.UnlessCondition(
                launch.substitutions.LaunchConfiguration('anonymous', default=defaults['anonymous'])
            ),
        ),

        # forces simulation=true when anonymous=true
        launch.actions.SetLaunchConfiguration(
            'simulation',
            value='true',
            condition=launch.conditions.IfCondition(
                launch.substitutions.LaunchConfiguration('anonymous', default=defaults['anonymous'])
            ),
        ),

        launch_ros.actions.Node(
            package='drive_api',
            executable='drive_api_node',
            # TODO: What does output do? Link to docs?
            output='screen',
            name=[launch.substitutions.LaunchConfiguration('node_name')],
            arguments=[
                [
                    'simulation=',
                    launch.substitutions.LaunchConfiguration('simulation', default=defaults['simulation'])
                ],
                [
                    'use_vesc=',
                    launch.substitutions.LaunchConfiguration('use_vesc', default=defaults['use_vesc'])
                ],
            ],
            parameters=[
                {'rate': launch.substitutions.LaunchConfiguration('rate', default=defaults['rate'])},
            ],
        ),

    })
