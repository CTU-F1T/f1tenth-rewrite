"""ROS node that converts angle received from Follow The Gap algorithm
(follow_the_gap_v0/follow_the_gap) to speed and heading angle.

 --------                          ----------------
| Follow |  /final_heading_angle  | Follow The Gap |  /drive_api/command   -----------
|  The   | ---------------------> |                | -------------------> | Drive-API |
|  Gap   |        Float32         |      Ride      |   drive_api_values    -----------
 --------                          ----------------

"""

from typing import List, Tuple, Any

# ROS 2 Python Client API
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import \
    ParameterDescriptor, \
    ParameterType, \
    Parameter, \
    SetParametersResult, \
    FloatingPointRange

# Computation engine (less memory consuming than numpy)
import math

# Message types
from std_msgs.msg import Bool
from drive_api_msgs.msg import DriveApiValues
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

        # Publishers
        self.pwm_pub = self.create_publisher(msg_type=DriveApiValues, topic='/drive_api/command', qos_profile=10)

        # Publishers for car constants (instead of CarControlData.msg)
        self.ltm_pub = self.create_publisher(msg_type=Float64, topic='ftg/turn/left/minF', qos_profile=1)
        self.lta_pub = self.create_publisher(msg_type=Float64, topic='ftg/turn/left/maxF', qos_profile=1)
        self.ltv_pub = self.create_publisher(msg_type=Float64, topic='ftg/turn/left/avoidF', qos_profile=1)

        self.rtm_pub = self.create_publisher(msg_type=Float64, topic='ftg/turn/right/minF', qos_profile=1)
        self.rta_pub = self.create_publisher(msg_type=Float64, topic='ftg/turn/right/maxF', qos_profile=1)
        self.rtv_pub = self.create_publisher(msg_type=Float64, topic='ftg/turn/right/avoidF', qos_profile=1)

        self.fsm_pub = self.create_publisher(msg_type=Float64, topic='ftg/speed/minF', qos_profile=1)
        self.fsa_pub = self.create_publisher(msg_type=Float64, topic='ftg/speed/maxF', qos_profile=1)
        self.fsv_pub = self.create_publisher(msg_type=Float64, topic='ftg/speed/avoidF', qos_profile=1)

        self.fas_pub = self.create_publisher(msg_type=Float64, topic='ftg/angle/switch', qos_profile=1)
        self.fah_pub = self.create_publisher(msg_type=Float64, topic='ftg/angle/hyster', qos_profile=1)
        self.fvs_pub = self.create_publisher(msg_type=Float64, topic='ftg/angle/switch_a', qos_profile=1)
        self.fvh_pub = self.create_publisher(msg_type=Float64, topic='ftg/angle/hyster_a', qos_profile=1)

        self.fal_pub = self.create_publisher(msg_type=Float64, topic='ftg/filter/alpha', qos_profile=1)

        # "global" variables

        # TODO: declare parameters and supply initial values

        # TODO: What's the real usage?
        #   Do we need set multiple parameters at once (atomically)?
        #   see https://answers.ros.org/question/358391/is-it-possible-to-set-multiple-parameters-in-ros2-from-the-command-line-using-ros2-param-set/

        # TODO: Did we use rqt_reconfigure (https://github.com/ros-visualization/rqt_reconfigure)?
        #   If we did, we might need to investigate rqt_reconfigure ROS 2 port state
        #   (it was ported to ROS 2 Dashing, see https://github.com/ros-visualization/rqt_reconfigure/issues)

        self.add_on_set_parameters_callback(self.reconfigure_callback)

        self.setup_parameters()

        # TODO: document mode values meaning and use meaningfully named constants for them
        self.mode = 0
        self.current_angle = self.VP_TURN_MAX_L
        self.current_drive = self.VP_SPEED_MAX
        # TODO: What is FILTER_ALPHA supposed to do?
        self.FILTER_ALPHA = 0.00

        self.timer = self.create_timer(2, self.timer_callback)

        pass

    def setup_parameters(self):

        # simplified parameters declaration
        # all parameters are of type ParameterType.PARAMETER_DOUBLE
        # and FloatingPointRange step is 0.0 (means no discrete step)
        # [name, description, default_value, from_value, to_value]
        params_simplified: List[Tuple[str, str, Any, Any, Any]] = [

            # PWM duty for speed (lesser number ~ faster)
            #   current meaning: AVOID is used during sharp turns,
            #                    MIN is used during other turns,
            #                    MAX elsewhere
            # self.VP_SPEED_MIN = 0.138522954  # ~ 9300
            # self.VP_SPEED_MAX = 0.158483034  # ~ 9350
            # self.VP_SPEED_AVOID = 0.098602794  # ~ 9200
            # TODO: there was a typo in speed_min / speed_max description (switched Level 1 / Level 2)
            #   Verify with @klapajar that it was really a typo and not an intention.
            ('speed_min', 'Level-1 speed', 0.138522954, 0.0, 0.3),
            ('speed_max', 'Level-2 speed', 0.158483034, 0.0, 0.3),
            ('speed_avoid', 'Level-3 speed', 0.098602794, 0.0, 0.3),

            # PWM duty for turning
            # TODO: Why the values in VP_* are NEGATIVE but parameters are POSITIVE?
            # # left
            # self.VP_TURN_MIN_L = -0.677419355  # ~ -2100
            # self.VP_TURN_MAX_L = -1  # ~ -3100
            # self.VP_TURN_AVOID_L = -1.419354839  # ~ -4400
            ('turn_l_min', 'Level-1 turn ratio (left)', 0.677419355, 0.0, 2.0),
            ('turn_l_max', 'Level-2 turn ratio (left)', 1.000000000, 0.0, 2.0),
            ('turn_l_avoid', 'Level-3 turn ratio (left)', 1.419354839, 0.0, 2.0),
            # # right
            # self.VP_TURN_MIN_R = -0.677419355  # ~ -2100
            # self.VP_TURN_MAX_R = -0.709677419  # ~ -2200
            # self.VP_TURN_AVOID_R = -1.290322581  # ~ -4000
            ('turn_r_min', 'Level-1 turn ratio (right)', 0.677419355, 0.0, 2.0),
            ('turn_r_max', 'Level-2 turn ratio (right)', 0.709677419, 0.0, 2.0),
            ('turn_r_avoid', 'Level-3 turn ratio (right)', 1.290322581, 0.0, 2.0),

            # Angle to switch between modes
            # self.ANGLE_SWITCH = math.radians(20.0)  # 20 degrees
            # self.ANGLE_HYSTER = math.radians(5)  # 5 degrees
            ('switch_l12', 'Level-1 to Level-2 angle switch', 20.0, 0.0, 90.0),
            ('hysteresis_l12', 'Level-1 to Level-2 angle hysteresis', 5.0, 0.0, 20.0),
            # self.ANGLE_SWITCH_A = math.radians(41.0)  # 41 degrees
            # self.ANGLE_HYSTER_A = math.radians(5.0)  # 5 degrees
            ('switch_l23', 'Level-2 to Level-3 angle switch', 41.0, 0.0, 90.0),
            ('hysteresis_l23', 'Level-2 to Level-3 angle hysteresis', 5.0, 0.0, 20.0),

        ]

        params_def: List[Tuple[str, Any, ParameterDescriptor]] = \
            [
                (
                    name,
                    value,
                    ParameterDescriptor(
                        type=ParameterType.PARAMETER_DOUBLE,
                        description=desc,
                        floating_point_range=[FloatingPointRange(
                            from_value=from_value,
                            to_value=to_value,
                            step=0.0,
                        )],
                    )
                )
                for name, desc, value, from_value, to_value in params_simplified
            ]

        self.declare_parameters(
            namespace='',
            parameters=params_def,
        )

        pass

    def timer_callback(self):

        hysteresis_l12_value = self.get_parameter('hysteresis_l12').value

        self.get_logger().info(f'hysteresis_l12 = {hysteresis_l12_value}')

        # my_new_param = rclpy.parameter.Parameter(
        #     'my_parameter',
        #     rclpy.Parameter.Type.STRING,
        #     'world'
        # )
        # all_new_parameters = [my_new_param]
        # self.set_parameters(all_new_parameters)

        pass

    def angle_callback(self, angle: Float32):
        """Computes speed and heading for received target angle from Follow The Gap algorithm.

        Arguments:
        angle -- angle (in radians) result from Follow The Gap algorithm, std_msgs.msg/Float32
        """

        if math.isnan(angle.data):
            return

        dv_msg = DriveApiValues()

        if abs(angle.data) < (self.ANGLE_SWITCH - self.ANGLE_HYSTER):
            self.mode = 0
        elif (
            (self.ANGLE_SWITCH + self.ANGLE_HYSTER) < abs(angle.data) < (self.ANGLE_SWITCH_A - self.ANGLE_HYSTER_A)
        ):
            self.mode = 1
        elif abs(angle.data) > (self.ANGLE_SWITCH_A + self.ANGLE_HYSTER_A):
            self.mode = 2

        # Go straight forward (high speed, low steering)
        if self.mode == 0:
            pwm_angle = (self.VP_TURN_MIN_L if angle.data < 0 else self.VP_TURN_MIN_R) * angle.data
            pwm_drive = self.VP_SPEED_MAX

        # Slow down before turning (low speed, high steering)
        elif self.mode == 1:
            pwm_angle = (self.VP_TURN_MAX_L if angle.data < 0 else self.VP_TURN_MAX_R) * angle.data
            pwm_drive = self.VP_SPEED_MIN

        # Slow down (a lot) before sharp turn (really low speed, really high steering)
        else:
            pwm_angle = (self.VP_TURN_AVOID_L if angle.data < 0 else self.VP_TURN_AVOID_R) * angle.data
            pwm_drive = self.VP_SPEED_AVOID

        # TODO: add note why this is commented out
        # dv_msg.pwm_angle = current_angle * (1 - FILTER_ALPHA) + pwm_angle * (FILTER_ALPHA)
        # dv_msg.pwm_drive = current_drive * (1 - FILTER_ALPHA) + pwm_drive * (FILTER_ALPHA)
        # current_drive =  current_drive * (1 - FILTER_ALPHA) + pwm_drive * (FILTER_ALPHA)

        dv_msg.velocity = pwm_drive
        dv_msg.forward = True

        if pwm_angle < 0:
            dv_msg.steering = -pwm_angle
            dv_msg.right = True
        else:
            dv_msg.steering = pwm_angle
            dv_msg.right = False

        if dv_msg.velocity > 0.3:  # Speed limiter
            dv_msg.velocity = 0.3

        if dv_msg.steering > 1.0:
            dv_msg.steering = 1.0

        self.pwm_pub.publish(dv_msg)

        pass

    def estop_callback(self, status: Bool):
        """Publishes internal constants each time the car is turned to autonomous mode.

        Arguments:
        status -- message triggering the autonomous mode, std_msgs.msg/Bool
        """

        # eStop == false --> that signals start
        if status.data is False:
            # Constants publishing
            msg_f6 = Float64()

            msg_f6.data = self.VP_TURN_MIN_L
            self.ltm_pub.publish(msg_f6)

            msg_f6.data = self.VP_TURN_MAX_L
            self.lta_pub.publish(msg_f6)

            msg_f6.data = self.VP_TURN_AVOID_L
            self.ltv_pub.publish(msg_f6)

            msg_f6.data = self.VP_TURN_MIN_R
            self.rtm_pub.publish(msg_f6)

            msg_f6.data = self.VP_TURN_MAX_R
            self.rta_pub.publish(msg_f6)

            msg_f6.data = self.VP_TURN_AVOID_R
            self.rtv_pub.publish(msg_f6)

            msg_f6.data = self.VP_SPEED_MIN
            self.fsm_pub.publish(msg_f6)

            msg_f6.data = self.VP_SPEED_MAX
            self.fsa_pub.publish(msg_f6)

            msg_f6.data = self.VP_SPEED_AVOID
            self.fsv_pub.publish(msg_f6)

            msg_f6.data = self.ANGLE_SWITCH
            self.fas_pub.publish(msg_f6)

            msg_f6.data = self.ANGLE_HYSTER
            self.fah_pub.publish(msg_f6)

            msg_f6.data = self.ANGLE_SWITCH_A
            self.fvs_pub.publish(msg_f6)

            msg_f6.data = self.ANGLE_HYSTER_A
            self.fvh_pub.publish(msg_f6)

            msg_f6.data = self.FILTER_ALPHA
            self.fal_pub.publish(msg_f6)

        pass

    pass

    def reconfigure_callback(self, parameters: List[Parameter]) -> SetParametersResult:
        """Called when any parameter changes

        Registered using self.add_on_set_parameters_callback(self.reconfigure_callback).
        Called for each parameter separately
        (unless multiple parameters are set using set_parameters_atomically).

        Before this callback is called, parameter value is validated against constraints (if any).
        If constraints validation fails, this callback will not be called at all.

        """

        self.get_logger().info('reconfigure_callback')

        ok = True

        # self.ANGLE_HYSTER = math.pi * (config["Hysteresis_L12"] / 180.0)

        # if we pass successful=False, parameter value will not be set
        # if parameter set was attempted using self.set_parameter*, node will exit with an error
        # if parameter set was attempted remotely, the remote caller is just passed the result
        # (failure and the optional configurable reason='why it was unsuccessful')
        return SetParametersResult(successful=True)

        # self.get_logger().info(f'''Reconfigure Request:
        #     Speed
        #         MAX: {config.Speed_max}
        #         MIN: {config.Speed_min}
        #         AVOID: {config.Speed_avoid}
        #     Turn-L
        #         MIN: {config.TurnL_min}
        #         MAX: {config.TurnL_max}
        #         AVOID: {config.TurnL_avoid}
        #     Turn-R
        #         MIN: {config.TurnR_min}
        #         MAX: {config.TurnR_max}
        #         AVOID: {config.TurnR_avoid}
        #     Switch 1-2
        #         Angle: {config.Switch_L12}
        #         Hysteresis: {config.Hysteresis_L12}
        #     Switch 2-3
        #         Angle: {config.Switch_L23}
        #         Hysteresis: {config.Hysteresis_L23}
        # ''')

        # rospy.loginfo(
        #     """Reconfigure Request: \n""" +
        #     """\tSpeed\n"""
        #     """\t\tMAX: {Speed_max}\n"""
        #     """\t\tMIN: {Speed_min}\n"""
        #     """\t\tAVOID: {Speed_avoid}\n"""
        #     """\tTurn-L\n"""
        #     """\t\tMIN: {TurnL_min}\n"""
        #     """\t\tMAX: {TurnL_max}\n"""
        #     """\t\tAVOID: {TurnL_avoid}\n"""
        #     """\tTurn-R\n"""
        #     """\t\tMIN: {TurnR_min}\n"""
        #     """\t\tMAX: {TurnR_max}\n"""
        #     """\t\tAVOID: {TurnR_avoid}\n"""
        #     """\tSwitch 1-2\n"""
        #     """\t\tAngle: {Switch_L12}\n"""
        #     """\t\tHysteresis: {Hysteresis_L12}\n"""
        #     """\tSwitch 2-3\n"""
        #     """\t\tAngle: {Switch_L23}\n"""
        #     """\t\tHysteresis: {Hysteresis_L23}"""
        #     .format(**config)
        # )

        self.VP_SPEED_MIN = config["Speed_min"]
        self.VP_SPEED_MAX = config["Speed_max"]
        self.VP_SPEED_AVOID = config["Speed_avoid"]

        self.VP_TURN_MAX_L = config["TurnL_max"]
        self.VP_TURN_MIN_L = config["TurnL_min"]
        self.VP_TURN_AVOID_L = config["TurnL_avoid"]

        self.VP_TURN_MAX_R = config["TurnR_max"]
        self.VP_TURN_MIN_R = config["TurnR_min"]
        self.VP_TURN_AVOID_R = config["TurnR_avoid"]

        self.ANGLE_SWITCH = math.pi * (config["Switch_L12"] / 180.0)
        self.ANGLE_HYSTER = math.pi * (config["Hysteresis_L12"] / 180.0)

        self.ANGLE_SWITCH_A = math.pi * (config["Switch_L23"] / 180.0)
        self.ANGLE_HYSTER_A = math.pi * (config["Hysteresis_L23"] / 180.0)

    pass


pass


def main(args=None):
    rclpy.init(args=args)

    ride_node = RideNode()

    rclpy.spin(ride_node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
