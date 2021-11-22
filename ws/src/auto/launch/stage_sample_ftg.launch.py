import launch
import launch.launch_description_sources
import launch_ros


def generate_launch_description():
    return launch.LaunchDescription([

        launch.actions.DeclareLaunchArgument(
            name='world',
            default_value='test_track',
            description='Name of a world within storage/stage/world',
        ),

        # TODO: rviz
        # TODO: map

        launch.actions.IncludeLaunchDescription(
            launch_description_source=launch.launch_description_sources.PythonLaunchDescriptionSource(
                launch_file_path=launch.substitutions.PathJoinSubstitution([
                    launch.substitutions.ThisLaunchFileDir(),
                    'stage.launch.py'
                ]),
            ),
            launch_arguments=[
                ('world', launch.substitutions.LaunchConfiguration(variable_name='world')),
            ],
        ),

        # TODO: Is it needed with stage_ros2?
        # <!-- Transformation -->
        # <node
        # 	pkg="tf" type="static_transform_publisher"
        # 	name="tf_base_laser_link"
        # 	args="0 0 0 0 0 0 base_laser_link /robot_0/base_laser_link 100"
        # />
        # <node
        # 	pkg="tf"
        # 	type="static_transform_publisher"
        # 	name="tf_odom_map"
        # 	args="0 0 0 0 0 0 odom map 100"
        # />

        # obstacle_substitution
        launch.actions.IncludeLaunchDescription(
            launch_description_source=launch.launch_description_sources.PythonLaunchDescriptionSource(
                launch_file_path=launch.substitutions.PathJoinSubstitution([
                    launch_ros.substitutions.FindPackageShare(package='obstacle_substitution'),
                    'start.launch.py'
                ]),
            ),
            launch_arguments=[
                ('remap', 'true'),
                ('/scan', '/stage_ros2/car/ranger_0'),
            ],
        ),

        # follow_the_gap_v0_ride
        launch.actions.IncludeLaunchDescription(
            launch_description_source=launch.launch_description_sources.PythonLaunchDescriptionSource(
                launch_file_path=launch.substitutions.PathJoinSubstitution([
                    launch_ros.substitutions.FindPackageShare(package='follow_the_gap_v0_ride'),
                    'start.launch.py'
                ]),
            ),
            launch_arguments=[],
        ),

        # drive_api
        launch_ros.actions.Node(
            package='drive_api',
            executable='drive_api_node',
            output='screen',
            name=launch.substitutions.AnonName(name='drive_api'),
            parameters=[
                launch.substitutions.PathJoinSubstitution([
                    launch_ros.substitutions.FindPackageShare(package='drive_api'),
                    'config',
                    'drive_api_node.yaml'
                ]),
                {
                    'run_mode': 'simulation',
                    'simulation.steering_modifier': 0.7,
                    'simulation.throttle_modifier': 7.26,
                },
            ],
            remappings=[
                ('/cmd_vel', '/stage_ros2/car/cmd_vel'),
            ],
        ),

    ])
