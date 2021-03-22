#!/usr/bin/env python
# follow_the_gap_ride.py
"""ROS node that converts angle received from Follow The Gap algorithm to speed and heading angle.

 --------                          ----------------
| Follow |  /final_heading_angle  | Follow The Gap |  /drive_api/command   -----------
|  The   | ---------------------> |                | -------------------> | Drive-API |
|  Gap   |        Float32         |      Ride      |   drive_api_values    -----------
 --------                          ----------------

"""
######################
# Imports & Globals
######################

# ROS python package
import rospy

# Computation engine (less memory consuming than numpy)
import math

# Dynamic reconfigure
import dynamic_reconfigure.server


# Dynamic reconfigure types
# ftgr
from follow_the_gap_v0.cfg import ftgrConfig


# Message types
# Bool
from std_msgs.msg import Bool
#: bool data

# drive_api_values
from drive_api.msg import drive_api_values
# Values for drive_api
#: float64 velocity        # allowed <0; 1>, positive values for forward direction
#: bool forward            # if true go forward, otherwise backwards
#: float64 steering        # allowed <0; 1>, positive values for right steering
#: bool right              # if true go right, otherwise left
# Note: Using negative values for velocity/steering will stop car/make it go straight.

# Float32
from std_msgs.msg import Float32
#: float32 data

# Float64
from std_msgs.msg import Float64
#: float64 data


# Publishers
pwm_pub = rospy.Publisher("/drive_api/command", drive_api_values, queue_size=10)

# Publishers for car constants (instead of CarControlData.msg)
ltm_pub = rospy.Publisher("ftg/turn/left/minF", Float64, queue_size=1)
lta_pub = rospy.Publisher("ftg/turn/left/maxF", Float64, queue_size=1)
ltv_pub = rospy.Publisher("ftg/turn/left/avoidF", Float64, queue_size=1)

rtm_pub = rospy.Publisher("ftg/turn/right/minF", Float64, queue_size=1)
rta_pub = rospy.Publisher("ftg/turn/right/maxF", Float64, queue_size=1)
rtv_pub = rospy.Publisher("ftg/turn/right/avoidF", Float64, queue_size=1)

fsm_pub = rospy.Publisher("ftg/speed/minF", Float64, queue_size=1)
fsa_pub = rospy.Publisher("ftg/speed/maxF", Float64, queue_size=1)
fsv_pub = rospy.Publisher("ftg/speed/avoidF", Float64, queue_size=1)

fas_pub = rospy.Publisher("ftg/angle/switch", Float64, queue_size=1)
fah_pub = rospy.Publisher("ftg/angle/hyster", Float64, queue_size=1)
fvs_pub = rospy.Publisher("ftg/angle/switch_a", Float64, queue_size=1)
fvh_pub = rospy.Publisher("ftg/angle/hyster_a", Float64, queue_size=1)

fal_pub = rospy.Publisher("ftg/filter/alpha", Float64, queue_size=1)


# Global variables
# PWM duty for speed (lesser number ~ faster)
# Currently - AVOID is used during sharp turns, MIN is used during other turns, MAX elsewhere
VP_SPEED_MIN = 0.138522954 # ~ 9300
VP_SPEED_MAX = 0.158483034 # ~ 9350
VP_SPEED_AVOID = 0.098602794 # ~ 9200

# PWM duty for turning
VP_TURN_MIN_L = -0.677419355 # ~ -2100
VP_TURN_MAX_L = -1 # ~ -3100
VP_TURN_AVOID_L = -1.419354839 # ~ -4400

VP_TURN_MIN_R = -0.677419355 # ~ -2100
VP_TURN_MAX_R = -0.709677419 # ~ -2200
VP_TURN_AVOID_R = -1.290322581 # ~ -4000

# Angle to switch between modes
ANGLE_SWITCH = math.pi * (20.0 / 180.0) # 20 degrees
ANGLE_HYSTER = math.pi * (5 / 180.0) # 5 degrees

ANGLE_SWITCH_A = math.pi * (41.0 / 180.0) # 40 degrees
ANGLE_HYSTER_A = math.pi * (5.0 / 180.0) # 5 degrees

mode = 0

current_angle = VP_TURN_MAX_L
current_drive = VP_SPEED_MAX
FILTER_ALPHA = 0.00


######################
# Callbacks
######################

def angle_callback(angle):
    """Computes speed and heading for received target angle from Follow The Gap algorithm.

    Arguments:
    angle -- angle (in radians) result from Follow The Gap algorithm, std_msgs.msg/Float32
    """
    global VP_SPEED_MIN, VP_SPEED_MAX, VP_SPEED_AVOID
    global VP_TURN_MAX_L, VP_TURN_MIN_L, VP_TURN_AVOID_L
    global VP_TURN_MAX_R, VP_TURN_MIN_R, VP_TURN_AVOID_R
    global ANGLE_SWITCH, ANGLE_HYSTER
    global ANGLE_SWITCH_A, ANGLE_HYSTER_A
    global mode
    global current_angle
    global current_drive

    if math.isnan(angle.data):
         return

    DVmsg = drive_api_values()

    if (abs(angle.data) < (ANGLE_SWITCH - ANGLE_HYSTER)):
        mode = 0
    elif (abs(angle.data) > (ANGLE_SWITCH + ANGLE_HYSTER) and abs(angle.data) < (ANGLE_SWITCH_A - ANGLE_HYSTER_A)):
        mode = 1
    elif (abs(angle.data) > (ANGLE_SWITCH_A + ANGLE_HYSTER_A)):
        mode = 2

    # Go straight forward (high speed, low steering)
    if mode == 0:
        pwm_angle = (VP_TURN_MIN_L if angle.data < 0 else VP_TURN_MIN_R) * angle.data
        pwm_drive = VP_SPEED_MAX

    # Slow down before turning (low speed, high steering)
    elif mode == 1:
        pwm_angle = (VP_TURN_MAX_L if angle.data < 0 else VP_TURN_MAX_R) * angle.data
        pwm_drive = VP_SPEED_MIN

    # Slow down (a lot) before sharp turn (really low speed, really high steering)
    else:
        pwm_angle = (VP_TURN_AVOID_L if angle.data < 0 else VP_TURN_AVOID_R) * angle.data
        pwm_drive = VP_SPEED_AVOID

    '''
    DVmsg.pwm_angle = current_angle * (1 - FILTER_ALPHA) + pwm_angle * (FILTER_ALPHA)
    DVmsg.pwm_drive = current_drive * (1 - FILTER_ALPHA) + pwm_drive * (FILTER_ALPHA)
    current_drive =  current_drive * (1 - FILTER_ALPHA) + pwm_drive * (FILTER_ALPHA)
    '''

    DVmsg.velocity = pwm_drive
    DVmsg.forward = True

    if pwm_angle < 0:
        DVmsg.steering = -pwm_angle
        DVmsg.right = True
    else:
        DVmsg.steering = pwm_angle
        DVmsg.right = False

    if DVmsg.velocity > 0.3: # Speed limiter
        DVmsg.velocity = 0.3

    if DVmsg.steering > 1.0:
        DVmsg.steering = 1.0

    pwm_pub.publish(DVmsg)


def estop_callback(status):
    """Publishes internal constants each time the car is turned to autonomous mode.

    Arguments:
    status -- message triggering the autonomous mode, std_msgs.msg/Bool
    """
    global VP_SPEED_MIN, VP_SPEED_MAX, VP_SPEED_AVOID
    global VP_TURN_MAX_L, VP_TURN_MIN_L, VP_TURN_AVOID_L
    global VP_TURN_MAX_R, VP_TURN_MIN_R, VP_TURN_AVOID_R
    global ANGLE_SWITCH, ANGLE_HYSTER
    global ANGLE_SWITCH_A, ANGLE_HYSTER_A

    if status.data is False:

        # Constants publishing
        msg_f6 = Float64()

        msg_f6.data = VP_TURN_MIN_L
        ltm_pub.publish(msg_f6)

        msg_f6.data = VP_TURN_MAX_L
        lta_pub.publish(msg_f6)

        msg_f6.data = VP_TURN_AVOID_L
        ltv_pub.publish(msg_f6)

        msg_f6.data = VP_TURN_MIN_R
        rtm_pub.publish(msg_f6)

        msg_f6.data = VP_TURN_MAX_R
        rta_pub.publish(msg_f6)

        msg_f6.data = VP_TURN_AVOID_R
        rtv_pub.publish(msg_f6)

        msg_f6.data = VP_SPEED_MIN
        fsm_pub.publish(msg_f6)

        msg_f6.data = VP_SPEED_MAX
        fsa_pub.publish(msg_f6)

        msg_f6.data = VP_SPEED_AVOID
        fsv_pub.publish(msg_f6)

        msg_f6.data = ANGLE_SWITCH
        fas_pub.publish(msg_f6)

        msg_f6.data = ANGLE_HYSTER
        fah_pub.publish(msg_f6)

        msg_f6.data = ANGLE_SWITCH_A
        fvs_pub.publish(msg_f6)

        msg_f6.data = ANGLE_HYSTER_A
        fvh_pub.publish(msg_f6)

        msg_f6.data = FILTER_ALPHA
        fal_pub.publish(msg_f6)


def reconf_callback(config, level):
    """Callback for maintaining dynamic reconfigure of this node."""
    global VP_SPEED_MIN, VP_SPEED_MAX, VP_SPEED_AVOID
    global VP_TURN_MAX_L, VP_TURN_MIN_L, VP_TURN_AVOID_L
    global VP_TURN_MAX_R, VP_TURN_MIN_R, VP_TURN_AVOID_R
    global ANGLE_SWITCH, ANGLE_HYSTER
    global ANGLE_SWITCH_A, ANGLE_HYSTER_A

    rospy.loginfo("""Reconfigure Request: \n""" +
                    """\tSpeed\n""" \
                        """\t\tMAX: {Speed_max}\n""" \
                        """\t\tMIN: {Speed_min}\n""" \
                        """\t\tAVOID: {Speed_avoid}\n""" \
                    """\tTurn-L\n""" \
                        """\t\tMIN: {TurnL_min}\n""" \
                        """\t\tMAX: {TurnL_max}\n""" \
                        """\t\tAVOID: {TurnL_avoid}\n""" \
                    """\tTurn-R\n""" \
                        """\t\tMIN: {TurnR_min}\n""" \
                        """\t\tMAX: {TurnR_max}\n""" \
                        """\t\tAVOID: {TurnR_avoid}\n""" \
                    """\tSwitch 1-2\n""" \
                        """\t\tAngle: {Switch_L12}\n""" \
                        """\t\tHysteresis: {Hysteresis_L12}\n""" \
                    """\tSwitch 2-3\n""" \
                        """\t\tAngle: {Switch_L23}\n""" \
                        """\t\tHysteresis: {Hysteresis_L23}""" \
                 .format(**config))

    VP_SPEED_MIN = config["Speed_min"]
    VP_SPEED_MAX = config["Speed_max"]
    VP_SPEED_AVOID = config["Speed_avoid"]

    VP_TURN_MAX_L = config["TurnL_max"]
    VP_TURN_MIN_L = config["TurnL_min"]
    VP_TURN_AVOID_L = config["TurnL_avoid"]

    VP_TURN_MAX_R = config["TurnR_max"]
    VP_TURN_MIN_R = config["TurnR_min"]
    VP_TURN_AVOID_R = config["TurnR_avoid"]

    ANGLE_SWITCH = math.pi * (config["Switch_L12"] / 180.0)
    ANGLE_HYSTER = math.pi * (config["Hysteresis_L12"] / 180.0)

    ANGLE_SWITCH_A = math.pi * (config["Switch_L23"] / 180.0)
    ANGLE_HYSTER_A = math.pi * (config["Hysteresis_L23"] / 180.0)

    return config


######################
# Functions
######################

def listener():
    """Initialize ROS node and create subscriber callback."""
    rospy.init_node("ftg_ride", anonymous=True)
    rospy.Subscriber("/final_heading_angle", Float32, angle_callback)
    rospy.Subscriber("/eStop", Bool, estop_callback)

    # Dynamic reconfigure
    srv = dynamic_reconfigure.server.Server(ftgrConfig, reconf_callback)

    rospy.spin()

if __name__ == "__main__":
    listener()
