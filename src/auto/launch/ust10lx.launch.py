import launch
import launch.actions
import launch.substitutions
import launch_ros.actions


# see https://docs.ros.org/en/rolling/Tutorials/Intermediate/Launch/Launch-system.html
# see https://github.com/ros2/launch/blob/rolling/launch/doc/source/architecture.rst
# see https://docs.ros.org/en/rolling/How-To-Guides/Node-arguments.html
def generate_launch_description():
    # note: Always use an order collection (e.g. List NOT Set),
    #       because the order here matters!
    return launch.LaunchDescription([

        launch.actions.DeclareLaunchArgument(
            name='/scan',
            default_value='/scan',
            description='Topic remapping',
        ),
        launch.actions.DeclareLaunchArgument(
            'remap',
            default_value='false',
            description='When true, use arguments above for remapping.',
        ),

        launch.actions.DeclareLaunchArgument(
            'ip_address',
            default_value='192.168.0.10',
            description='IP address of LiDAR',
        ),

        launch.actions.DeclareLaunchArgument(
            'ip_port',
            default_value='10940',
            description='IP port number of LiDAR',
        ),

        launch_ros.actions.Node(
            package='urg_node',
            executable='urg_node_driver',
            output='screen',
            name='lidar_ust10lx',
            parameters=[
                {
                    'angle_max': 3.14,
                    'angle_min': -3.14,
                    'ip_address': launch.substitutions.LaunchConfiguration('ip_address'),
                    'ip_port': launch.substitutions.LaunchConfiguration('ip_port'),
                    'serial_baud': 115200,
                    'laser_frame_id': 'laser',
                    'calibrate_time': False,
                    'default_user_latency': 0.0,
                    'diagnostics_tolerance': 0.05,
                    'diagnostics_window_time': 5.0,
                    'error_limit': 4,
                    'get_detailed_status': False,
                    'publish_intensity': False,
                    'publish_multiecho': False,
                    'cluster': 1,
                    'skip': 0,
                },
            ],
        ),

    ])
