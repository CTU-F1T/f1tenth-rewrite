#!/usr/bin/env python
# launch_components.py
"""Launch components selected via argument.

This ROS node represents an unified way to launch components according to their
type. It is mandatory to have configuration file loaded to the ROS Parameter
Server.

This script is just quick extension of 'sensor_launcher'.
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

# Global variables
REQUIREALL = False


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
    global REQUIREALL

    # Obtain arguments without remaps
    args = rospy.myargv(argv=sys.argv)

    if len(args) < 2:
        rospy.logerr("Missing arguments.")
        return

    rospy.init_node("component_launcher", anonymous=True)

    # Obtain parameters from the server, ...
    params = []

    layers = ["/perception/sensors", "/perception/preprocessing", "/perception/recognition", "/decision_and_control",
              "/vehicle_platform", "/custom"]

    for l in layers:
        # ... sorted, only "the interesting part" ...
        for p in sorted(rospy.get_param(l, {}).items(), key=lambda x: x[0]):
            params.append(p[1])

    if len(params) == 0:
        rospy.logerr("Unable to receive parameters from the server.")
        return

    # ... and remove all invalid parameters
    for p in params:
        if ('type' not in p or '!name' not in p) and 'model' not in p:
            del p

    # Create list for all the launch files
    launch_files = list()

    # Go through all of the arguments
    # But at first, delete the file name
    del args[0]

    for a in args:
        found = 0

        for p in params:
            if p.get("!name", p.get("type")) == a:
                rospy.loginfo("Detected valid argument: %s", p)

                # Unroll parameters
                if "!params" in p:
                    for param in p.get("!params").values():
                        p[param.get("key")] = param.get("value")

                # Require other targets
                if "!require" in p:
                    for param in p.get("!require"):
                        args.append(param)

                # Fail on not finding packages (only affects next args)
                if "!requireall" in p:
                    REQUIREALL = str(p.get("!requireall")).lower() == "true"

                # Skip when model is not there
                if "model" in p:
                    try:
                        # Get absolute path for the launch file
                        launch_files.append((roslaunch_lunar.rlutil.resolve_launch_arguments(
                            [p.get("model"), p.get("!launch", "start") + ".launch"])[0], dictToParams(p)))
                    except roslaunch_lunar.core.RLException:
                        rospy.logfatal("Detected corrupted configuration: Package or launch file doesn't exist.")

                        if REQUIREALL:
                            raise

                params.remove(p)
                break
        else:
            rospy.logerr("Obtained invalid argument: %s", a)

            if REQUIREALL:
                rospy.logfatal("An error occurred during reading the arguments. Aborting execution.")
                return

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
