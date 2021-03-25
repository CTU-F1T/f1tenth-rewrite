# F1/10 @ CTU (ROS 2 rewrite)


## Status

* decision_and_control
    * **follow_the_gap_v0** (C++, launch files)
        * C++ node
    * **follow_the_gap_v0_ride** (Python)
        * extracted from follow_the_gap_v0 package
* messages
    * **command_msgs** (IDL)
    * **f1tenth_race** (IDL)
    * **obstacle_msgs** (IDL)
    * **plan_msgs** (IDL)
    * **trajectory** (IDL)
    * **vesc_msgs** (IDL)
* vehicle_platform
    * **drive_api** (Python, launch files)
    * **drive_api_msgs**
        * extracted from drive_api package,
          so that drive_api could be pure Python package
    * **teensy** (IDL, launch files)
        * TODO: launch file, rosserial_python


## Notes


### Pure Python Package cannot contain messages

ROS 2 _pure_ Python package cannot contain IDL definitions (messages, services)
_(at least for now, it may be supported in the future)_.
It is possible use ament_cmake with Python code but I am not sure
if `colcon build --symlink-install` option works then.
And also it is unnecessary verbose.  


### Launch files are not symlinked

Normally `generate_launch_description` functions should be placed
in their respective launch files (e.g. `launch/start.launch.py`).
**BUT** `--symlink-install` option does not work for launch files:
* see https://github.com/colcon/colcon-core/issues/407
* see https://github.com/ros2/launch/issues/187

We have several possible **workarounds**:
1. _(The currently used one, seems to be the best one.)_ Place launch definitions to package dir and include them
    from respective launch files. This way rebuild is needed only once. 
    The idea comes from https://github.com/colcon/colcon-core/issues/169#issuecomment-531517276
2. Another possible workaround is mentioned here (although it seems rather impractical) https://github.com/ros2/launch/issues/187#issuecomment-468667696.
3. We could create a script to manually symlink them after build (while doing dev). But that would be tiresome and error-prone.


### Others

* **There is no "global parameter server" in ROS 2**
    * see https://discourse.ros.org/t/ros2-global-parameter-server-status/10114
    * see https://github.com/fujitatomoya/ros2_persist_parameter_server
