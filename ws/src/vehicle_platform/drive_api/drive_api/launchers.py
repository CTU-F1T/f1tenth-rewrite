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
    # note: Always use an order collection (e.g. List NOT Set),
    #       because the order here matters!
    return launch.LaunchDescription([

        launch.actions.DeclareLaunchArgument(
            name='/drive_api/command',
            default_value='/drive_api/command',
            description='Topic remapping',
        ),
        launch.actions.DeclareLaunchArgument(
            name='/drive_pwm',
            default_value='/drive_pwm',
            description='Topic remapping',
        ),
        launch.actions.DeclareLaunchArgument(
            name='/cmd_vel',
            default_value='/cmd_vel',
            description='Topic remapping',
        ),
        launch.actions.DeclareLaunchArgument(
            name='/eStop',
            default_value='/eStop',
            description='Topic remapping',
        ),
        launch.actions.DeclareLaunchArgument(
            'remap',
            default_value='false',
            description='When true, use arguments above for remapping.',
        ),

        launch.actions.DeclareLaunchArgument(
            'anonymous',
            default_value='false',
            description='When true, run the node as anonymous (generate random name).',
        ),
        launch.actions.DeclareLaunchArgument(
            'simulation',
            default_value='false',
            description=(
                'When true, run the node in simulation mode. '
                'If anonymous=true then it the simulation=true is forced no matter what.'
            ),
        ),
        launch.actions.DeclareLaunchArgument(
            'use_vesc',
            default_value='false',
            # TODO: Maybe change "rather than" to "instead of"?
            description='When true speed is published directly to VESC rather than to Teensy.',
        ),
        launch.actions.DeclareLaunchArgument(
            'rate',
            default_value='10',
            description='Publish rate of the Drive-API messages.',
        ),
        launch.actions.DeclareLaunchArgument(
            'speed_modifier',
            default_value='1.0',
            description='Speed modifier for simulation mode ONLY.',
            condition=launch.conditions.IfCondition(
                launch.substitutions.LaunchConfiguration('simulation')
            ),
        ),
        launch.actions.DeclareLaunchArgument(
            'steer_modifier',
            default_value='1.0',
            description='Steering modifier for simulation mode ONLY.',
            condition=launch.conditions.IfCondition(
                launch.substitutions.LaunchConfiguration('simulation')
            ),
        ),

        launch.actions.SetLaunchConfiguration(
            'node_name',
            value=launch.substitutions.AnonName(name='drive_api'),
            condition=launch.conditions.IfCondition(
                launch.substitutions.LaunchConfiguration('anonymous')
            ),
        ),
        launch.actions.SetLaunchConfiguration(
            'node_name',
            value='drive_api',
            condition=launch.conditions.UnlessCondition(
                launch.substitutions.LaunchConfiguration('anonymous')
            ),
        ),

        # forces simulation=true when anonymous=true
        launch.actions.SetLaunchConfiguration(
            'simulation',
            value='true',
            condition=launch.conditions.IfCondition(
                launch.substitutions.LaunchConfiguration('anonymous')
            ),
        ),

        launch.actions.LogInfo(
            msg=['node_name=', launch.substitutions.LaunchConfiguration('node_name')],
        ),

        launch_ros.actions.Node(
            package='drive_api',
            executable='drive_api_node',
            output='screen',
            name=[launch.substitutions.LaunchConfiguration('node_name')],
            arguments=[
                [
                    'simulation=',
                    launch.substitutions.LaunchConfiguration('simulation')
                ],
                [
                    'use_vesc=',
                    launch.substitutions.LaunchConfiguration('use_vesc')
                ],
            ],
            parameters=[
                {'rate': launch.substitutions.LaunchConfiguration('rate')},
                # TODO: pass speed_modifier and steer_modifier iff simulation=true
                {'speed_modifier': launch.substitutions.LaunchConfiguration('speed_modifier')},
                {'steer_modifier': launch.substitutions.LaunchConfiguration('steer_modifier')},
            ],
            remappings=[
                # TODO: only if remap=true
                ('/drive_api/command', launch.substitutions.LaunchConfiguration('/drive_api/command')),
                ('/drive_pwm', launch.substitutions.LaunchConfiguration('/drive_pwm')),
                ('/cmd_vel', launch.substitutions.LaunchConfiguration('/cmd_vel')),
                ('/eStop', launch.substitutions.LaunchConfiguration('/eStop')),
            ],
        ),

    ])
