import launch
import launch.launch_description_sources
import launch_ros


def generate_launch_description():
    return launch.LaunchDescription([

        launch.actions.DeclareLaunchArgument(
            name='config',
            description='vehicle config',
            # default_value='tx2-auto-3.yaml',
        ),

        launch.actions.SetLaunchConfiguration(
            name='config_file',
            value=launch.substitutions.PathJoinSubstitution([
                launch_ros.substitutions.FindPackageShare(package='auto'),
                'config',
                launch.substitutions.LaunchConfiguration(variable_name='config'),
            ]),
        ),

        launch.actions.LogInfo(
            msg=['using config file = ', launch.substitutions.LaunchConfiguration(variable_name='config_file')],
        ),

        # obstacle_substitution
        launch.actions.IncludeLaunchDescription(
            launch_description_source=launch.launch_description_sources.PythonLaunchDescriptionSource(
                launch_file_path=launch.substitutions.PathJoinSubstitution([
                    launch_ros.substitutions.FindPackageShare(package='obstacle_substitution'),
                    'start.launch.py'
                ]),
            ),
            launch_arguments=[],
        ),

        # lidar
        launch.actions.IncludeLaunchDescription(
            launch_description_source=launch.launch_description_sources.PythonLaunchDescriptionSource(
                launch_file_path=launch.substitutions.PathJoinSubstitution([
                    launch_ros.substitutions.FindPackageShare(package='auto'),
                    'launch',
                    'ust10lx.launch.py'
                ]),
            ),
            launch_arguments=[],
        ),

        # static transformations
        launch.actions.IncludeLaunchDescription(
            launch_description_source=launch.launch_description_sources.PythonLaunchDescriptionSource(
                launch_file_path=launch.substitutions.PathJoinSubstitution([
                    launch_ros.substitutions.FindPackageShare(package='auto'),
                    'launch',
                    'auto.static_tf.launch.py',
                ]),
            ),
            launch_arguments=[],
        ),

        launch_ros.actions.Node(
            executable='static_transform_publisher',
            name='tf_laser',
            package='tf2_ros',
            # output='screen',
            arguments=[
                # x y z qx qy qz qw
                '0', '0', '0', '0', '0', '0',
                # frame_id child_frame_id
                'horizontal_laser_link', 'laser',
            ],
        ),

        # odometry
        launch_ros.actions.Node(
            package='vesc_ackermann',
            executable='vesc_to_odom_node',
            output='screen',
            name='vesc_to_odom',
            parameters=[
                launch.substitutions.LaunchConfiguration(variable_name='config_file')
            ],
        ),

        # pwm_to_steer
        launch_ros.actions.Node(
            package='pwm_to_steer',
            executable='pwm_to_steer_node',
            output='screen',
            name='pwm_to_steer',
            parameters=[
                launch.substitutions.LaunchConfiguration(variable_name='config_file')
            ],
        ),

        # vesc_driver
        launch_ros.actions.Node(
            package='vesc_driver',
            executable='vesc_driver_node',
            output='screen',
            name='vesc',
            parameters=[
                launch.substitutions.LaunchConfiguration(variable_name='config_file')
            ],
            remappings=[
                ('/sensors/servo_position_command', '/not_used/servo_position_command'),
            ],
        ),

        # teensy_drive
        launch_ros.actions.Node(
            package='teensy_drive',
            executable='teensy_drive',
            output='screen',
            name='teensy_drive',
            parameters=[
                launch.substitutions.LaunchConfiguration(variable_name='config_file')
            ],
        ),

    ])
