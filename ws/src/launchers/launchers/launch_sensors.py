#!/usr/bin/env python
# launch_sensors.py
"""Launch sensors selected via argument.

This ROS node represents an unified way to launch sensors according to their
type. It is mandatory to have configuration file loaded to the ROS Parameter
Server.
"""
######################
# Imports & Globals
######################

# ROS python package
import rospy

# System
import sys

# Ported roslaunch from ros-lunar
# This version supports argument for Python roslaunch-api.
import roslaunch_lunar


######################
# Functions
######################

def dictToParams(d):
    """Converts dictionary with parameters to roslaunch-params-styled list.

    Arguments:
    d -- parameters, dictionary

    Returns:
    params -- parameters, list
    """
    params = list()

    for (k, v) in d.items():
        params.append(str(k) + ":=" + str(v))

    return params


def main():
    """Starts a node which launches all sensors that are specified in the
    arguments.
    """

    # Obtain arguments without remaps
    args = rospy.myargv(argv=sys.argv)

    if len(args) < 2:
        rospy.logerr("Missing arguments.")
        return

    rospy.init_node("sensor_launcher", anonymous=True)

    # Obtain parameters from the server, ...
    core_params = rospy.get_param("/perception/sensors", [])

    if len(core_params) == 0:
        rospy.logerr("Unable to receive parameters from the server.")
        return

    # ... sorted, only "the interesting part" ...
    params = [p[1] for p in sorted(core_params.items(), key=lambda x: x[0])]

    # ... and remove all invalid parameters
    for p in params:
        if 'type' not in p and 'model' not in p:
            del p

    # Create list for all the launch files
    launch_files = list()

    # Go through all of the arguments
    # But at first, delete the file name
    del args[0]

    for a in args:
        found = 0

        for p in params:
            if p.get("type") == a:
                rospy.loginfo("Detected valid argument: %s", p)

                try:
                    # Get absolute path for the launch file
                    launch_files.append((roslaunch_lunar.rlutil.resolve_launch_arguments(
                        [p.get("model"), "start.launch"])[0], dictToParams(p)))
                except roslaunch_lunar.core.RLException:
                    rospy.logfatal("Detected corrupted configuration: Package or launch file doesn't exist.")

                params.remove(p)
                break
        else:
            rospy.logerr("Obtained invalid argument: %s", a)

    if len(launch_files) == 0:
        rospy.logwarn("No valid launch file found.")
        return

    # Initialize roslaunch (with some additional magic)
    uuid = roslaunch_lunar.rlutil.get_or_generate_uuid(None, False)
    roslaunch_lunar.configure_logging(uuid)
    launch = roslaunch_lunar.scriptapi.ROSLaunch()
    launch.parent = roslaunch_lunar.parent.ROSLaunchParent(uuid, launch_files)
    launch.parent.start()

    # Keep the nodes running...
    try:
        launch.spin()
    # ... unless Ctrl+C is pressed.
    finally:
        launch.stop()


if __name__ == '__main__':
    main()
