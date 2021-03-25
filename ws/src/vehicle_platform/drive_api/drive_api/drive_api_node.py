"""Sends PWM data to the car controller.

This module (ROS node) is divided into two parts using threads. Main thread is used as
importable package to provide low level access to the car speed and steering. Secondary
thread is doing the publishing part as it is publishing to /drive_pwm, /cmd_vel, and
/commands/motor/speed topics.

Available variables:
Direction -- FORWARD / BACKWARD / LEFT / RIGHT, enum
ControlMode -- LEGACY / JOINT / METRIC / ANGULAR, enum

Available functions:
init(file_dest) -- load the config file and initialize the node on secondary thread
setSpeed(speed, direction, control_mode|LEGACY) -- sets the speed with given direction
    - setForwardSpeed(speed, [control_mode]) = setSpeed(speed, FORWARD, control_mode|LEGACY)
    - setBackwardSpeed(speed, [control_mode]) = setSpeed(speed, BACKWARD, control_mode|LEGACY)
stop() -- stops the car
setSteer(steer, direction, control_mode|LEGACY) -- sets the steer with given direction
    - setLeftSteer(steer, [control_mode]) = setSteer(steer, LEFT, control_mode|LEGACY)
    - setRightSteer(steer, [control_mode]) = setSteer(steer, RIGHT, control_mode|LEGACY)
resetSteer() -- resets the steering angle of the wheels

Usage:
 1) Import this ROS node as a package to another node.
 2) Call .init(file_dest) function, where 'file_dest' is a location of configuration
    file.
 3) Now you can use other API functions.

When launched directly, this node acts as ordinary ROS node. It listens to topics
/drive_api/command (see drive_api.msg/drive_api_values message for more) and /command
(see commands_msgs.msg/CommandArrayStamped message for more), which are used to give
velocity / steering commands to the Drive-API.

Drive-API can publish several types of messages depending on selected settings:
BASIC (simulation=False, use_vesc=False)
 In BASIC mode, Drive-API is publishing to /drive_pwm topic which is subscribed by Teensy board.
 Both speed and steering is sent this way, specified in PWM duty.
BASIC+VESC (simulation=False, use_vesc=True)
 In BASIC+VESC mode, Drive-API is publishing steering to /drive_pwm topic (to be handled by Teensy),
 and speed to /commands/motor/speed topic, which is subscribed by VESC.
SIMULATION (simulation=True)
 In SIMULATION mode, Drive-API is publishing both speed and steering to /cmd_vel topic which
 is used as a standard for various simulators.

For controlling the speed and steering, following control modes are available:
LEGACY
 In LEGACY control mode, speed/steering value is specified in range from 0 to 1 (both included)
 and a direction. Boundary values have unusual meaning -- 0 represents lowest possible
 speed/steering, 1 represents highest possible speed/steering. Using values lower than 0 results
 in stopping the vehicle/resetting the steering. Values larger than 1 are ignored.
JOINT
 In JOINT control mode, speed/steering value is specified in range from -1 to 1 (both included).
 Boundary values have ordinary meaning -- 1 is highest speed (largest left steering) and -1 is
 highest backward speed (largest right steering). Requesting value 0 results in stopping
 the vehicle/resetting the steering. Values outside the range are ignored.
METRIC
 In METRIC control mode, only speed can be passed. It is specified in unlimited range in [m.s^-1].
 An attempt to pass steering in this control mode results in an error.
ANGULAR
 In ANGULAR control mode, only steering can be passed. It is specified in [rad] and in a range
 limited by 'SERVO_LEFT_MAX' and 'SERVO_RIGHT_MAX'. LEGACY steering parameters are required while
 using this control mode. An attempt to pass speed in this control mode results in an error.
"""
######################
# Imports & Globals
######################

from typing import List, Tuple, Any, Dict

# ROS 2 Python Client API
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import \
    ParameterDescriptor, \
    ParameterType, \
    Parameter, \
    SetParametersResult, \
    FloatingPointRange

# PI constant
from math import pi

# Handling arguments
import sys

# Threading
import threading

# JSON
import json

# Enum support
from enum import Enum

# Message types
# Bool
from std_msgs.msg import Bool

# bool data

try:
    # Command
    from command_msgs.msg import Command
    # Command message
    #: string command
    #: command_msgs/CommandParameter[] parameters

    # CommandArrayStamped
    from command_msgs.msg import CommandArrayStamped
    # CommandArrayStamped message
    # List of commands, with a timestamp
    #: Header header
    #: command_msgs/Command[] commands

    # CommandParameter
    from command_msgs.msg import CommandParameter

    # CommandParameter message
    #: string parameter
    #: float64 value

    USE_COMMANDS = True
except ImportError:
    print('Unable to find Command* messages. Subscriber will be disabled.', file=sys.stderr)
    USE_COMMANDS = False

# drive_values
from teensy.msg import DriveValues

# int16 pwm_drive           # PWM duty cycle (0-100%) corresponds to (0-65535) interval
# int16 pwm_angle

try:
    # drive_api_values
    from drive_api_msgs.msg import DriveApiValues

    # float64 velocity          # allowed <0; 1>, positive values for forward direction
    # bool forward              # if true go forward, otherwise backwards
    # float64 steering          # allowed <0; 1>, positive values for right steering
    # bool right                # if true go right, otherwise left
    # Note: Using negative values for velocity/steering will stop car/make it go straight.

    USE_DAPI_COMMANDS = True
except ImportError:
    print('Unable to find drive_api_values message. Subscriber will be disabled.', file=sys.stderr)
    USE_DAPI_COMMANDS = False

# Float64
from std_msgs.msg import Float64
#: float64 data

# Twist
from geometry_msgs.msg import Twist
# This expresses velocity in free space broken into its linear and angular parts.
# Vector3  linear
# Vector3  angular

# Vector3
from geometry_msgs.msg import Vector3

# This represents a vector in free space.
# float64 x
# float64 y
# float64 z

# Publishers
pub = None  # rospy.Publisher('drive_pwm', drive_values, queue_size=1)
pub_cmd_vel = None  # rospy.Publisher('cmd_vel', Twist, queue_size=1)
pub_vesc = None  # rospy.Publisher('commands/motor/speed', Float64, queue_size=1)

# Global variables
msg = DriveValues()
msg_vesc = Float64()
msg_cmd_vel = Twist()
lock = threading.Lock()
constants = {}
eStop = True
run_mode = None


# Direction enum
class Direction(Enum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3


# RunMode enum
class RunMode(Enum):
    BASIC = 0
    BASIC_VESC = 1
    SIMULATION = 2


# Mode enum
class ControlMode(Enum):
    LEGACY = 0
    JOINT = 1
    METRIC = 2
    ANGULAR = 3


######################
# Functions (API)
######################

def setSpeed(speed, direction, control_mode=ControlMode.LEGACY):
    """Set the car speed at desired value with given direction.

    Arguments:
    speed -- speed, float
    direction -- direction of movement, Direction enum

    Returns:
    success -- False if encountered errors, otherwise True

    ControlMode option:
    LEGACY -- speed is given from 0 to 1 (both included), direction
              is specified via 'direction'
    JOINT -- speed is given from -1 to 1 (both included), direction
             is expected to be 'FORWARD', otherwise it is flipped
    METRIC -- speed is given from unlimited range in [m.s^-1], direction
              is expected to be 'FORWARD' otherwise it is flipped
    ANGULAR -- not supported
    """
    global lock
    global constants
    global eStop
    global run_mode

    # Check eStop
    if eStop:
        return False

    # Check the constants dict
    if not constants:
        return False

    # Check the control mode
    if not isinstance(control_mode, ControlMode):
        return False

    # Continue according to the selected mode
    if control_mode == ControlMode.LEGACY:
        # Check the speed
        if speed < 0 or speed > 1:
            return False

        # Check the direction
        if not isinstance(direction, Direction):
            return False

        # Set corresponding variable
        if direction == Direction.BACKWARD:
            with lock:
                if run_mode == RunMode.BASIC:
                    if ('BACKWARD_MIN' in constants) and ('BACKWARD' in constants):
                        msg.pwm_drive = int(constants['BACKWARD_MIN'] + speed * constants['BACKWARD'])
                    else:
                        rospy.logerr("Unable to compute PWM speed because 'BACKWARD_MAX' / 'BACKWARD_MIN' is not set.")
                        return False

                elif run_mode == RunMode.SIMULATION:
                    msg_cmd_vel.linear.x = -speed * constants['SIM_SPEEDUP']

                elif run_mode == RunMode.BASIC_VESC:
                    if 'ERPM_MAX' in constants:
                        msg_vesc.data = -speed * constants['ERPM_MAX']
                    else:
                        rospy.logerr("Unable to compute VESC speed because 'ERPM_MAX' is not set.")
                        return False

        elif direction == Direction.FORWARD:
            with lock:
                if run_mode == RunMode.BASIC:
                    if ('FORWARD_MIN' in constants) and ('FORWARD' in constants):
                        msg.pwm_drive = int(constants['FORWARD_MIN'] + speed * constants['FORWARD'])
                    else:
                        rospy.logerr("Unable to compute PWM speed because 'FORWARD_MAX' / 'FORWARD_MIN' is not set.")
                        return False

                elif run_mode == RunMode.SIMULATION:
                    msg_cmd_vel.linear.x = speed * constants['SIM_SPEEDUP']

                elif run_mode == RunMode.BASIC_VESC:
                    if 'ERPM_MAX' in constants:
                        msg_vesc.data = speed * constants['ERPM_MAX']
                    else:
                        rospy.logerr("Unable to compute VESC speed because 'ERPM_MAX' is not set.")
                        return False
        else:
            return False

    elif control_mode == ControlMode.JOINT:
        # Check the speed
        if speed < -1 or speed > 1:
            return False

        # Check the direction
        if not isinstance(direction, Direction):
            return False

        # Set corresponding variable
        if speed == 0:
            stop()
        elif (
            (direction == Direction.FORWARD and speed > 0)
            or (direction == Direction.BACKWARD and speed < 0)
        ):
            with lock:
                if run_mode == RunMode.BASIC:
                    if ('FORWARD_MIN' in constants) and ('FORWARD' in constants):
                        msg.pwm_drive = int(constants['FORWARD_MIN'] + abs(speed) * constants['FORWARD'])
                    else:
                        rospy.logerr("Unable to compute PWM speed because 'FORWARD_MAX' / 'FORWARD_MIN' is not set.")
                        return False

                elif run_mode == RunMode.SIMULATION:
                    msg_cmd_vel.linear.x = abs(speed) * constants['SIM_SPEEDUP']

                elif run_mode == RunMode.BASIC_VESC:
                    if 'ERPM_MAX' in constants:
                        msg_vesc.data = abs(speed) * constants['ERPM_MAX']
                    else:
                        rospy.logerr("Unable to compute VESC speed because 'ERPM_MAX' is not set.")
                        return False

        elif (
            (direction == Direction.BACKWARD and speed > 0)
            or (direction == Direction.FORWARD and speed < 0)
        ):
            with lock:
                if run_mode == RunMode.BASIC:
                    if ('BACKWARD_MIN' in constants) and ('BACKWARD' in constants):
                        msg.pwm_drive = int(constants['BACKWARD_MIN'] + abs(speed) * constants['BACKWARD'])
                    else:
                        rospy.logerr("Unable to compute PWM speed because 'BACKWARD_MAX' / 'BACKWARD_MIN' is not set.")
                        return False

                elif run_mode == RunMode.SIMULATION:
                    msg_cmd_vel.linear.x = -abs(speed) * constants['SIM_SPEEDUP']

                elif run_mode == RunMode.BASIC_VESC:
                    if 'ERPM_MAX' in constants:
                        msg_vesc.data = -abs(speed) * constants['ERPM_MAX']
                    else:
                        rospy.logerr("Unable to compute VESC speed because 'ERPM_MAX' is not set.")
                        return False
        else:
            return False

    elif control_mode == ControlMode.METRIC:
        # Check run modes
        if run_mode == RunMode.BASIC:
            rospy.logerr("Unsupported run mode for data in 'METRIC' format.")
            return False

        # Check the direction
        if not isinstance(direction, Direction):
            return False

        # Set corresponding variable
        if speed == 0:
            stop()
        elif run_mode == RunMode.SIMULATION:
            with lock:
                msg_cmd_vel.linear.x = speed
        elif 'TO_ERPM' in constants:
            with lock:
                msg_vesc.data = speed * constants["TO_ERPM"]
        else:
            rospy.logerr("Unable to compute VESC speed because 'TO_ERPM' is not set.")
            return False

    else:
        rospy.logerr("Unsupported data format.")
        return False

    return True


def setForwardSpeed(speed, control_mode=ControlMode.LEGACY):
    """Trampoline function for 'setSpeed()'."""
    return setSpeed(speed, Direction.FORWARD, control_mode)


def setBackwardSpeed(speed, control_mode=ControlMode.LEGACY):
    """Trampoline function for 'setSpeed()'."""
    return setSpeed(speed, Direction.BACKWARD, control_mode)


def stop():
    """Send a calm state speed. It means that the car stops."""
    global msg
    global lock
    global constants

    # Check the constants dict
    if not constants:
        return False

    with lock:
        if 'CALM_SPEED' in constants:
            msg.pwm_drive = constants['CALM_SPEED']

        msg_cmd_vel.linear.x = 0
        msg_vesc.data = 0

    return True


def setSteer(steer, direction, control_mode=ControlMode.LEGACY):
    """Set the car steering at desired value with given direction.

    Arguments:
    steer -- steer, float
    direction -- direction of movement, Direction enum

    Returns:
    success -- False if encountered errors, otherwise True

    ControlMode option:
    LEGACY -- steer is given from 0 to 1 (both included), direction
              is specified via 'direction'
    JOINT -- steer is given from -1 to 1 (both included), direction
             is expected to be 'LEFT', otherwise it is flipped
    METRIC -- not supported
    ANGULAR -- steer is given from unlimited range in [rad], but limited
               by 'SERVO_LEFT_MAX' and 'SERVO_RIGHT_MAX' parameters,
               direction is expected to be 'LEFT', otherwise it is flipped
    """
    global lock
    global constants
    global eStop
    global run_mode

    # Check eStop
    if eStop:
        return False

    # Check the constants dict
    if not constants:
        return False

    # Check the control mode
    if not isinstance(control_mode, ControlMode):
        return False

    # Continue according to the select mode
    if control_mode == ControlMode.LEGACY:
        # Check the steering
        if steer < 0 or steer > 1:
            return False

        # Check the direction
        if not isinstance(direction, Direction):
            return False

        # Simulation mode
        if run_mode == RunMode.SIMULATION:
            if steer == 0:
                resetSteer()
            else:
                msg_cmd_vel.angular.z = steer * constants['SIM_STEERUP'] * (
                    1.0 if direction == Direction.LEFT else -1.0)

            return True

        # Set corresponding variable
        if direction == Direction.LEFT:
            with lock:
                if ('LEFT_MIN' in constants) and ('LEFT' in constants):
                    msg.pwm_angle = int(constants['LEFT_MIN'] + steer * constants['LEFT'])
                else:
                    rospy.logerr("Unable to compute PWM steer because 'LEFT_MAX' / 'LEFT_MIN' is not set.")
                    return False

        elif direction == Direction.RIGHT:
            with lock:
                if ('RIGHT_MIN' in constants) and ('RIGHT' in constants):
                    msg.pwm_angle = int(constants['RIGHT_MIN'] + steer * constants['RIGHT'])
                else:
                    rospy.logerr("Unable to compute PWM steer because 'RIGHT_MAX' / 'RIGHT_MIN' is not set.")
                    return False

        else:
            return False

    elif control_mode == ControlMode.JOINT:
        # Check the steering
        if steer < -1 or steer > 1:
            return False

        # Check the direction
        if not isinstance(direction, Direction):
            return False

        # Simulation mode
        if run_mode == RunMode.SIMULATION:
            if steer == 0:
                resetSteer()
            else:
                msg_cmd_vel.angular.z = abs(steer) * constants['SIM_STEERUP'] * (
                    1.0 if direction == Direction.LEFT else -1.0)

            return True

        # Set corresponding variable
        if steer == 0:
            resetSteer()
        elif (
            (direction == Direction.LEFT and steer > 0)
            or (direction == Direction.RIGHT and steer < 0)
        ):
            with lock:
                if ('LEFT_MIN' in constants) and ('LEFT' in constants):
                    msg.pwm_angle = int(constants['LEFT_MIN'] + abs(steer) * constants['LEFT'])
                else:
                    rospy.logerr("Unable to compute PWM steer because 'LEFT_MAX' / 'LEFT_MIN' is not set.")
                    return False

        elif (
            (direction == Direction.RIGHT and steer > 0)
            or (direction == Direction.LEFT and steer < 0)
        ):
            with lock:
                if ('RIGHT_MIN' in constants) and ('RIGHT' in constants):
                    msg.pwm_angle = int(constants['RIGHT_MIN'] + abs(steer) * constants['RIGHT'])
                else:
                    rospy.logerr("Unable to compute PWM steer because 'RIGHT_MAX' / 'RIGHT_MIN' is not set.")
                    return False

        else:
            return False

    elif control_mode == ControlMode.ANGULAR:
        # Check the direction
        if not isinstance(direction, Direction):
            return False

        # Simulation mode
        if run_mode == RunMode.SIMULATION:
            if steer == 0:
                resetSteer()
            else:
                msg_cmd_vel.angular.z = steer if direction == Direction.LEFT else -steer

            return True

        # Check the required parameters
        if ('SERVO_LEFT_MAX' not in constants) or ('SERVO_RIGHT_MAX' not in constants):
            rospy.logerr("Unable to compute PWM steer because 'SERVO_LEFT_MAX' / 'SERVO_RIGHT_MAX' is not set.")
            return False

        if steer == 0:
            resetSteer()
        elif (
            (direction == Direction.LEFT and steer > 0)
            or (direction == Direction.RIGHT and steer < 0)
        ):
            with lock:
                if ('LEFT_MIN' in constants) and ('LEFT' in constants):
                    msg.pwm_angle = int(
                        constants['LEFT_MIN'] + min(abs(steer) / constants['SERVO_LEFT_MAX'], 1.0) * constants['LEFT'])
                else:
                    rospy.logerr("Unable to compute PWM steer because 'LEFT_MAX' / 'LEFT_MIN' is not set.")
                    return False

        elif (
            (direction == Direction.RIGHT and steer > 0)
            or (direction == Direction.LEFT and steer < 0)
        ):
            with lock:
                if ('RIGHT_MIN' in constants) and ('RIGHT' in constants):
                    msg.pwm_angle = int(
                        constants['RIGHT_MIN'] + min(abs(steer) / constants['SERVO_RIGHT_MAX'], 1.0) * constants[
                            'RIGHT'])
                else:
                    rospy.logerr("Unable to compute PWM steer because 'RIGHT_MAX' / 'RIGHT_MIN' is not set.")
                    return False

    else:
        rospy.logerr("Unsupported data format.")
        return False

    return True


def setLeftSteer(steer, control_mode=ControlMode.LEGACY):
    """Trampoline function for 'setSteer()'."""
    return setSteer(steer, Direction.LEFT, control_mode)


def setRightSteer(steer, control_mode=ControlMode.LEGACY):
    """Trampoline function for 'setSteer()'."""
    return setSteer(steer, Direction.RIGHT, control_mode)


def resetSteer():
    """Send a calm state steer. It means that the car turns the wheels to go straight."""
    global msg
    global lock
    global constants

    # Check the constants dict
    if not constants:
        return False

    with lock:
        if 'CALM_STEER' in constants:
            msg.pwm_angle = constants['CALM_STEER']

        msg_cmd_vel.angular.z = 0


######################
# Callbacks (ROS)
######################

def api_callback(data):
    """Obtain requested speed/steering from ROS topic.

    Arguments:
    data -- structure received on topic /drive_api/command, defined by drive_api_values
    """

    if data.velocity < 0:
        stop()
    elif data.forward:
        setForwardSpeed(data.velocity)
    else:
        setBackwardSpeed(data.velocity)

    if data.steering < 0:
        resetSteer()
    elif data.right:
        setRightSteer(data.steering)
    else:
        setLeftSteer(data.steering)


def command_callback(data):
    """Obtain requested speed/steering from ROS topic.

    Arguments:
    data -- structure received on topic /command, defined by CommandArrayStamped
    """

    # Check eStop
    if eStop:
        return False

    if len(data.commands) > 0:
        for c in data.commands:
            if c.command == "speed" and len(c.parameters) > 0:
                for p in c.parameters:
                    success = False

                    # LEGACY
                    if p.parameter == "forward":
                        if p.value < 0:
                            success = stop()
                        else:
                            success = setForwardSpeed(p.value, ControlMode.LEGACY)
                    elif p.parameter == "backward":
                        if p.value < 0:
                            success = stop()
                        else:
                            success = setBackwardSpeed(p.value, ControlMode.LEGACY)
                    # JOINT
                    elif p.parameter == "norm" or p.parameter == "joint":
                        success = setForwardSpeed(p.value, ControlMode.JOINT)
                    # METRIC
                    elif p.parameter == "metric" or p.parameter == "m/s":
                        success = setForwardSpeed(p.value, ControlMode.METRIC)

                    if not success:
                        rospy.logerr("Unable to process '%s' parameter: %s (%f)" % (c.command, p.parameter, p.value))
                        continue

                    break
            elif c.command == "steer" and len(c.parameters) > 0:
                for p in c.parameters:
                    success = False

                    # LEGACY
                    if p.parameter == "left":
                        if p.value < 0:
                            success = stop()
                        else:
                            success = setLeftSteer(p.value, ControlMode.LEGACY)
                    elif p.parameter == "right":
                        if p.value < 0:
                            success = stop()
                        else:
                            success = setRightSteer(p.value, ControlMode.LEGACY)
                    # JOINT
                    elif p.parameter == "norm" or p.parameter == "joint":
                        success = setLeftSteer(p.value, ControlMode.JOINT)
                    # ANGULAR
                    elif p.parameter == "deg":
                        success = setLeftSteer(p.value / 180.0 * pi, ControlMode.ANGULAR)
                    elif p.parameter == "rad":
                        success = setLeftSteer(p.value, ControlMode.ANGULAR)

                    if not success:
                        rospy.logerr("Unable to process '%s' parameter: %s (%f)" % (c.command, p.parameter, p.value))
                        continue

                    break
            else:
                rospy.logerr("Unable to process command: %s" % c.command)


def estop_callback(data):
    """Obtain emergency stop message from ROS topic.

    This message can be also used to stop/start simulation.

    Arguments:
    data -- structure received on topic /eStop, defined by std_msgs.msg/Bool
    """
    global eStop

    eStop = data.data

    if eStop:
        # Set calm states
        stop()
        resetSteer()


######################
# Functions (ROS)
######################

def publish():
    """Publish currently stored variables.

    Note: Used when not in simulation mode. Only 'pub' is used.
    """
    global msg
    global msg_cmd_vel
    global lock

    with lock:
        pub.publish(msg)
        # pub_cmd_vel.publish(msg_cmd_vel)


def publish_with_vesc():
    """Publish currently stored variables.

    Note: Used when not in simulation mode and when 'use_vesc' is True.
    Note: Mutually exclusive with 'publish'.
    """
    global constants
    global msg
    global msg_vesc
    global lock

    with lock:
        pub.publish(DriveValues(pwm_drive=constants['CALM_SPEED'], pwm_angle=msg.pwm_angle))
        pub_vesc.publish(msg_vesc)


def publish_sim():
    """Publish currently stored variables.

    Note: Used when in simulation mode. Only 'pub_cmd_vel' is used.
    """
    global msg_cmd_vel
    global lock

    with lock:
        pub_cmd_vel.publish(msg_cmd_vel)


def publisher(init_node=True, create_callbacks=False, anonymous=False, simulation=False, use_vesc=False):
    """Initialize ROS node and publish variables."""
    global pub
    global pub_vesc
    global pub_cmd_vel
    global run_mode

    # For this node, only one should be running. Disable signals for threaded publisher
    # Allow multiple nodes for simulation purposes.
    if init_node:
        rospy.init_node('drive_api', anonymous=anonymous, disable_signals=True)

    # Create callbacks if requested
    if create_callbacks:
        if USE_DAPI_COMMANDS:
            rospy.Subscriber("/drive_api/command", DriveApiValues, api_callback, queue_size=1)

        if USE_COMMANDS:
            rospy.Subscriber("/command", CommandArrayStamped, command_callback, queue_size=1)

    # Register to 2-way /eStop
    rospy.Subscriber("/eStop", Bool, estop_callback, queue_size=1)

    # Action modifiers for simulation (can be received only after 'init_node' as they are private).
    constants['SIM_SPEEDUP'] = rospy.get_param("~speed_modifier") * 1.0 if rospy.has_param("~speed_modifier") else 1.0
    constants['SIM_STEERUP'] = rospy.get_param("~steer_modifier") * 1.0 if rospy.has_param("~steer_modifier") else 1.0

    # Function rate() creates rate object to loop at the desired rate in Hz
    if rospy.has_param("~rate"):
        rate = rospy.Rate(rospy.get_param("~rate"))
        rospy.loginfo("Setting the publish rate of the node to the requested %d Hz." % rospy.get_param("~rate"))
    else:
        rate = rospy.Rate(10)
        rospy.loginfo("Publishing the messages with default rate 10 Hz.")

    # Create VESC publisher
    if use_vesc:
        pub_vesc = rospy.Publisher('commands/motor/speed', Float64, queue_size=1)

    # Propagate run mode
    run_mode = RunMode.BASIC_VESC if use_vesc else (RunMode.BASIC if not simulation else RunMode.SIMULATION)

    # Create SIMULATION publisher
    if run_mode == RunMode.SIMULATION:
        pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    else:
        pub = rospy.Publisher('drive_pwm', DriveValues, queue_size=1)

    # Attach publish function
    if use_vesc and ('CALM_SPEED' in constants) and ('CALM_STEER' in constants):
        pub_function = publish_with_vesc
    elif simulation:
        pub_function = publish_sim
    elif ('CALM_SPEED' in constants) and ('CALM_STEER' in constants):
        pub_function = publish
    else:
        rospy.logerr("Unable to attach a publish function. Shutting down the node.")
        return

    # Function is_shutdown() reacts to exit flag (Ctrl+C, etc.)
    while not rospy.is_shutdown():
        pub_function()

        try:
            rate.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException as e:
            rospy.logwarn("It is possible that publishing consistency has been violated now.")


def constFromParamServer(kwargs):
    """Obtains parameters from the ROS Parameter server.

    Note: When parameters are not properly loaded, node ends.
    """
    global constants

    try:
        params = rospy.get_param("parameters")
    except:
        print("Error during receiving constants from Parameter Server. Are they set?", file=sys.stderr)

    # Reach for constants separately as we can work with one set only.
    try:
        constants['CALM_SPEED'] = params.get("long").get("calm")
        constants['FORWARD_MIN'] = params.get("long").get("forward").get("min")
        constants['FORWARD_MAX'] = params.get("long").get("forward").get("max")
        constants['BACKWARD_MIN'] = params.get("long").get("backward").get("min")
        constants['BACKWARD_MAX'] = params.get("long").get("backward").get("max")

        # Compute rest of the constants
        constants['FORWARD'] = constants['FORWARD_MAX'] - constants['FORWARD_MIN']
        constants['BACKWARD'] = constants['BACKWARD_MAX'] - constants['BACKWARD_MIN']

        # Set calm state
        stop()
    except:
        print("Error during receiving constants for PWM driving. It will be disabled.", file=sys.stderr)

    try:
        constants['CALM_STEER'] = params.get("lat").get("calm")
        constants['LEFT_MIN'] = params.get("lat").get("left").get("min")
        constants['LEFT_MAX'] = params.get("lat").get("left").get("max")
        constants['RIGHT_MIN'] = params.get("lat").get("right").get("min")
        constants['RIGHT_MAX'] = params.get("lat").get("right").get("max")

        # Compute rest of the constants
        constants['LEFT'] = constants['LEFT_MAX'] - constants['LEFT_MIN']
        constants['RIGHT'] = constants['RIGHT_MAX'] - constants['RIGHT_MIN']

        # Set calm state
        resetSteer()
    except:
        print("Error during receiving constants for PWM steering. It will be disabled.", file=sys.stderr)

    # VESC control constants
    if kwargs.get("use_vesc"):
        try:
            constants['ERPM_MAX'] = 1.0 * params.get("motor").get("erpm_max")
        except:
            try:
                constants['ERPM_MAX'] = (params.get("motor").get("poles") / 2.0) * params.get("motor").get("back_emf") \
                                        * params.get("battery").get("cells") * params.get("battery").get("voltage")
            except:
                print(
                    "Unable to receive and compute 'ERPM_MAX' parameter for 'LEGACY'"
                    "and 'JOINT' control modes.They will be disabled.",
                    file=sys.stderr,
                )

        try:
            constants['TO_ERPM'] = 1.0 * params.get("motor").get("to_erpm")
        except:
            try:
                constants['TO_ERPM'] = (params.get("motor").get("poles") / 2.0) \
                                       * (
                                           1.0 * params.get("differential").get("spur")
                                           / params.get("motor").get("pinion")
                                       ) \
                                       * (
                                           1.0 * params.get("differential").get("ring")
                                           / params.get("differential").get("pinion")
                                       ) \
                                       / (2.0 * pi * params.get("wheels").get("radius")) \
                                       * 60
            except:
                print(
                    "Unable to receive and compute 'TO_ERPM' parameter for 'METRIC' control mode. It will be disabled.",
                    file=sys.stderr
                )

        if ('ERPM_MAX' not in constants) and ('TO_ERPM' not in constants):
            print(
                "Unable to receive any constants for VESC control. Reverting to original method.",
                file=sys.stderr,
            )
            kwargs["use_vesc"] = False

    # Angular control constants
    try:
        constants['SERVO_LEFT_MAX'] = params.get("servo").get("left").get("max") / 180.0 * pi
        constants['SERVO_RIGHT_MAX'] = params.get("servo").get("right").get("max") / 180.0 * pi
    except:
        pass

    publisher(True, True, **kwargs)


######################
# Functions (Thread)
######################

def init(file_dest, init_node=True, use_vesc=False):
    """Initialize python script and create threaded publisher."""
    ok = load_constants(file_dest, use_vesc)

    if not ok:
        print('Encountered error during initialization. Exiting.', file=sys.stderr)
        return

    t = threading.Thread(target=publisher, args=[init_node, False, False, False, use_vesc])
    t.start()


######################
# Functions (Python)
######################

def load_constants(file_dest, use_vesc):
    """Open and load constants from a file."""
    global constants

    # TODO: Check the file
    with open(file_dest) as f:
        constants = json.loads(f.read())

    # Compute rest of the constants
    if 'CALM_SPEED' in constants:
        constants['FORWARD'] = constants['FORWARD_MAX'] - constants['FORWARD_MIN']
        constants['BACKWARD'] = constants['BACKWARD_MAX'] - constants['BACKWARD_MIN']

    if 'CALM_STEER' in constants:
        constants['LEFT'] = constants['LEFT_MAX'] - constants['LEFT_MIN']
        constants['RIGHT'] = constants['RIGHT_MAX'] - constants['RIGHT_MIN']

    if use_vesc:
        if 'ERPM_MAX' not in constants:
            try:
                constants['ERPM_MAX'] = constants['MOTOR_BACK_EMF'] \
                                        * constants['BATTERY_CELLS'] \
                                        * constants['BATTERY_CELL_VOLTAGE']
            except:
                print(
                    "Unable to read and compute 'ERPM_MAX' constant, disabling control modes 'LEGACY' and 'JOINT'.",
                    file=sys.stderr,
                )

        if 'TO_ERPM' not in constants:
            try:
                constants['TO_ERPM'] = (constants['MOTOR_POLES'] / 2.0) \
                                       * (1.0 * constants['DIFF_MOTOR_SPUR'] / constants['MOTOR_DIFF_PINION']) \
                                       * (1.0 * constants['WHEEL_DIFF_RING'] / constants['DIFF_WHEEL_PINION']) \
                                       / (2.0 * pi * constants['WHEEL_RADIUS']) \
                                       * 60
            except:
                print(
                    "Unable to read and compute 'TO_ERPM' constant, disabling control mode 'METRIC'.",
                    file=sys.stderr,
                )

    # Set calm states
    stop()
    resetSteer()

    return not use_vesc or (use_vesc and (constants.has_key['ERPM_MAX'] or constants.has_key['TO_ERPM']))


if __name__ == '__main__':
    args = [a.lower() for a in rospy.myargv(argv=sys.argv)]

    # Slightly ugly, but efficient solution
    # Force simulation mode when running anonymous node
    anonymous = "anonymous=true" in args
    simulation = "simulation=true" in args or anonymous
    use_vesc = "use_vesc=true" in args

    constFromParamServer({"anonymous": anonymous, "simulation": simulation, "use_vesc": use_vesc})
