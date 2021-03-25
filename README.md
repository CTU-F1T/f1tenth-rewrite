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

* ROS 2 _pure_ Python package cannot contain IDL definitions (messages, services)
    _(at least for now, it may be supported in the future)_.
    It is possible use ament_cmake with Python code but I am not sure
    if `colcon build --symlink-install` option works then.
    And also it is unnecessary verbose.  
