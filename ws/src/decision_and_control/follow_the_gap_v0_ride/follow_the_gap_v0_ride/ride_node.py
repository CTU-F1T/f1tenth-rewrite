"""

ROS node that converts angle received from Follow The Gap algorithm (follow_the_gap_v0/follow_the_gap)
to speed and heading angle.

 --------                          ----------------
| Follow |  /final_heading_angle  | Follow The Gap |  /drive_api/command   -----------
|  The   | ---------------------> |                | -------------------> | Drive-API |
|  Gap   |        Float32         |      Ride      |   drive_api_values    -----------
 --------                          ----------------

"""

# ROS 2 Python Client API
import rclpy
from rclpy.node import Node

# Computation engine (less memory consuming than numpy)
import math

# Message types
from std_msgs.msg import Bool
# drive_api_values
from drive_api.msg import drive_api_values
# Values for drive_api
#: float64 velocity        # allowed <0; 1>, positive values for forward direction
#: bool forward            # if true go forward, otherwise backwards
#: float64 steering        # allowed <0; 1>, positive values for right steering
#: bool right              # if true go right, otherwise left
# Note: Using negative values for velocity/steering will stop car/make it go straight.
from std_msgs.msg import Float32
from std_msgs.msg import Float64


class RideNode(Node):

    def __init__(self):
        super().__init__('follow_the_gap_ride')

        self.angle_subscription = self.create_subscription(
            msg_type=Float32,
            topic='/final_heading_angle',
            callback=self.angle_callback,
            qos_profile=1,  # TODO: better QoS settings?
        )

        self.estop_subscription = self.create_subscription(
            msg_type=Bool,
            topic='/eStop',
            callback=self.estop_callback,
            qos_profile=1,  # TODO: better QoS settings?
        )

        pass

    def add_three_ints_callback(self, request: AddThreeInts.Request, response: AddThreeInts.Response):
        response.sum = request.a + request.b + request.c
        self.get_logger().info('Incoming request\na: %d b: %d c: %d' %
                               (request.a, request.b, request.c))
        return response

    def angle_callback(self, angle):
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
            pwm_angle = (VP_TURN_MIN_L if angle.data <
                         0 else VP_TURN_MIN_R) * angle.data
            pwm_drive = VP_SPEED_MAX

        # Slow down before turning (low speed, high steering)
        elif mode == 1:
            pwm_angle = (VP_TURN_MAX_L if angle.data <
                         0 else VP_TURN_MAX_R) * angle.data
            pwm_drive = VP_SPEED_MIN

        # Slow down (a lot) before sharp turn (really low speed, really high steering)
        else:
            pwm_angle = (VP_TURN_AVOID_L if angle.data <
                         0 else VP_TURN_AVOID_R) * angle.data
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

        if DVmsg.velocity > 0.3:  # Speed limiter
            DVmsg.velocity = 0.3

        if DVmsg.steering > 1.0:
            DVmsg.steering = 1.0

        pwm_pub.publish(DVmsg)

    def estop_callback(self, status):
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

    def reconf_callback(self, config, level):
        """Callback for maintaining dynamic reconfigure of this node."""
        global VP_SPEED_MIN, VP_SPEED_MAX, VP_SPEED_AVOID
        global VP_TURN_MAX_L, VP_TURN_MIN_L, VP_TURN_AVOID_L
        global VP_TURN_MAX_R, VP_TURN_MIN_R, VP_TURN_AVOID_R
        global ANGLE_SWITCH, ANGLE_HYSTER
        global ANGLE_SWITCH_A, ANGLE_HYSTER_A

        rospy.loginfo("""Reconfigure Request: \n""" +
                      """\tSpeed\n"""
                      """\t\tMAX: {Speed_max}\n"""
                      """\t\tMIN: {Speed_min}\n"""
                      """\t\tAVOID: {Speed_avoid}\n"""
                      """\tTurn-L\n"""
                      """\t\tMIN: {TurnL_min}\n"""
                      """\t\tMAX: {TurnL_max}\n"""
                      """\t\tAVOID: {TurnL_avoid}\n"""
                      """\tTurn-R\n"""
                      """\t\tMIN: {TurnR_min}\n"""
                      """\t\tMAX: {TurnR_max}\n"""
                      """\t\tAVOID: {TurnR_avoid}\n"""
                      """\tSwitch 1-2\n"""
                      """\t\tAngle: {Switch_L12}\n"""
                      """\t\tHysteresis: {Hysteresis_L12}\n"""
                      """\tSwitch 2-3\n"""
                      """\t\tAngle: {Switch_L23}\n"""
                      """\t\tHysteresis: {Hysteresis_L23}"""
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


def main(args=None):
    rclpy.init(args=args)

    ride_node = RideNode()

    rclpy.spin(ride_node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
