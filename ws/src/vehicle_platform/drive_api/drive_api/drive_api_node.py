"""Sends PWM data to the car controller.

This module (ROS node) is doing the publishing part
as it is publishing to /drive_pwm, /cmd_vel, and /commands/motor/speed topics.

It listens to topics /drive_api/command (see drive_api.msg/drive_api_values message for more)
and /command (see commands_msgs.msg/CommandArrayStamped message for more),
which are used to give velocity / steering commands to the Drive-API.

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

import sys
import threading

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

from enum import Enum

# Message types
from std_msgs.msg import Bool
from std_msgs.msg import Float64

try:
    from command_msgs.msg import Command
    from command_msgs.msg import CommandArrayStamped
    from command_msgs.msg import CommandParameter

    USE_COMMANDS = True
except ImportError:
    print('Unable to find Command* messages. Subscriber will be disabled.', file=sys.stderr)
    USE_COMMANDS = False

from teensy.msg import DriveValues

try:
    from drive_api_msgs.msg import DriveApiValues

    USE_DAPI_COMMANDS = True
except ImportError:
    print('Unable to find drive_api_values message. Subscriber will be disabled.', file=sys.stderr)
    USE_DAPI_COMMANDS = False

# Twist expresses velocity in free space broken into its linear and angular parts:
#   Vector3 linear
#   Vector3 angular
# Vector3 represents a vector in free space:
#   float64 x
#   float64 y
#   float64 z
from geometry_msgs.msg import Twist


class Direction(Enum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3


class RunMode(Enum):
    BASIC = 0
    BASIC_VESC = 1
    SIMULATION = 2


class ControlMode(Enum):
    LEGACY = 0
    JOINT = 1
    METRIC = 2
    ANGULAR = 3


class InitError(Exception):
    pass


class DriveApiNode(Node):

    def __init__(self, create_callbacks=False, simulation=False, use_vesc=False):
        super().__init__(node_name='drive_api')

        self.get_logger().info(f'initializing, simulation={simulation}, use_vesc={use_vesc}')

        # publishers
        self.pub = None  # self.create_publisher(msg_type=DriveValues, topic='drive_pwm', qos_profile=1)
        self.pub_cmd_vel = None  # self.create_publisher(msg_type=Twist, topic='cmd_vel', qos_profile=1)
        self.pub_vesc = None  # self.create_publisher(msg_type=Float64, topic='commands/motor/speed', qos_profile=1)

        #  variables
        self.msg = DriveValues()
        self.msg_vesc = Float64()
        self.msg_cmd_vel = Twist()
        self.lock = threading.Lock()
        self.constants = {}
        self.eStop = True
        self.run_mode = None

        self.dapi_cmds_subscription = None
        self.cmds_subscription = None

        # Create callbacks if requested
        if create_callbacks:
            if USE_DAPI_COMMANDS:
                self.dapi_cmds_subscription = self.create_subscription(
                    msg_type=DriveApiValues,
                    topic='/drive_api/command',
                    callback=self.api_callback,
                    qos_profile=1,
                )

            if USE_COMMANDS:
                self.cmds_subscription = self.create_subscription(
                    msg_type=CommandArrayStamped,
                    topic='/command',
                    callback=self.command_callback,
                    qos_profile=1,
                )

        # Register to 2-way /eStop
        self.estop_subscription = self.create_subscription(
            msg_type=Bool,
            topic='/eStop',
            callback=self.estop_callback,
            qos_profile=1,
        )

        # TODO: load from params
        # Action modifiers for simulation (can be received only after 'init_node' as they are private).
        # self.constants['SIM_SPEEDUP'] = rospy.get_param('~speed_modifier') * 1.0 if rospy.has_param(
        #     '~speed_modifier') else 1.0
        # self.constants['SIM_STEERUP'] = rospy.get_param('~steer_modifier') * 1.0 if rospy.has_param(
        #     '~steer_modifier') else 1.0

        # TODO: https://answers.ros.org/question/358343/rate-and-sleep-function-in-rclpy-library-for-ros2/
        # TODO: https://nicolovaligi.com/concurrency-and-parallelism-in-ros1-and-ros2-application-apis.html
        # TODO: https://yuzhangbit.github.io/tools/several-ways-of-writing-a-ros-node/

        # Function rate() creates rate object to loop at the desired rate in Hz
        # if rospy.has_param('~rate'):
        #     rate = rospy.Rate(rospy.get_param('~rate'))
        #     self.get_logger().info(
        #         f'Setting the publish rate of the node to the requested {rospy.get_param("~rate"):d} Hz.'
        #     )
        # else:
        #     rate = rospy.Rate(10)
        #     self.get_logger().info('Publishing the messages with default rate 10 Hz.')
        # TODO: configurable rate (via param)
        self.pub_rate = self.create_rate(frequency=10)

        # Create VESC publisher
        if use_vesc:
            self.create_publisher(msg_type=Float64, topic='commands/motor/speed', qos_profile=1)

        # Propagate run mode
        run_mode = RunMode.BASIC_VESC if use_vesc else (RunMode.BASIC if not simulation else RunMode.SIMULATION)

        # Create SIMULATION publisher
        if run_mode == RunMode.SIMULATION:
            self.pub_cmd_vel = self.create_publisher(msg_type=Twist, topic='cmd_vel', qos_profile=1)
        else:
            self.pub = self.create_publisher(msg_type=DriveValues, topic='drive_pwm', qos_profile=1)

        # Attach publish function
        if use_vesc and ('CALM_SPEED' in self.constants) and ('CALM_STEER' in self.constants):
            pub_function = self.publish_with_vesc
        elif simulation:
            pub_function = self.publish_sim
        elif ('CALM_SPEED' in self.constants) and ('CALM_STEER' in self.constants):
            pub_function = self.publish
        else:
            self.get_logger().error('Unable to attach a publish function. Shutting down the node.')
            raise InitError('Unable to attach a publish function. Shutting down the node.')

        # Function is_shutdown() reacts to exit flag (Ctrl+C, etc.)
        # TODO: rewrite
        while rclpy.ok():
            pub_function()
            self.pub_rate.sleep()

        pass

    def setup_parameters(self):

        # TODO

        pass

    def publish(self):
        """Publish currently stored variables.

        Note: Used when not in simulation mode. Only 'pub' is used.
        """

        with self.lock:
            self.pub.publish(self.msg)
            # pub_cmd_vel.publish(msg_cmd_vel)

    def publish_with_vesc(self):
        """Publish currently stored variables.

        Note: Used when not in simulation mode and when 'use_vesc' is True.
        Note: Mutually exclusive with 'publish'.
        """

        with self.lock:
            self.pub.publish(
                DriveValues(pwm_drive=self.constants['CALM_SPEED'], pwm_angle=self.msg.pwm_angle))
            self.pub_vesc.publish(self.msg_vesc)

    def publish_sim(self):
        """Publish currently stored variables.

        Note: Used when in simulation mode. Only 'pub_cmd_vel' is used.
        """

        with self.lock:
            self.pub_cmd_vel.publish(self.msg_cmd_vel)

    def set_speed(self, speed, direction, control_mode=ControlMode.LEGACY):
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

        # Check eStop
        if self.eStop:
            return False

        # Check the self.constants dict
        if not self.constants:
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
                with self.lock:
                    if self.run_mode == RunMode.BASIC:
                        if ('BACKWARD_MIN' in self.constants) and ('BACKWARD' in self.constants):
                            self.msg.pwm_drive = int(
                                self.constants['BACKWARD_MIN'] + speed * self.constants['BACKWARD'])
                        else:
                            self.get_logger().info(
                                'Unable to compute PWM speed because \'BACKWARD_MAX\' / \'BACKWARD_MIN\' is not set.'
                            )
                            return False

                    elif self.run_mode == RunMode.SIMULATION:
                        self.msg_cmd_vel.linear.x = -speed * self.constants['SIM_SPEEDUP']

                    elif self.run_mode == RunMode.BASIC_VESC:
                        if 'ERPM_MAX' in self.constants:
                            self.msg_vesc.data = -speed * self.constants['ERPM_MAX']
                        else:
                            self.get_logger().info(
                                'Unable to compute VESC speed because \'ERPM_MAX\' is not set.'
                            )
                            return False

            elif direction == Direction.FORWARD:
                with self.lock:
                    if self.run_mode == RunMode.BASIC:
                        if ('FORWARD_MIN' in self.constants) and ('FORWARD' in self.constants):
                            self.msg.pwm_drive = int(self.constants['FORWARD_MIN'] + speed * self.constants['FORWARD'])
                        else:
                            self.get_logger().info(
                                'Unable to compute PWM speed because \'FORWARD_MAX\' / \'FORWARD_MIN\' is not set.'
                            )
                            return False

                    elif self.run_mode == RunMode.SIMULATION:
                        self.msg_cmd_vel.linear.x = speed * self.constants['SIM_SPEEDUP']

                    elif self.run_mode == RunMode.BASIC_VESC:
                        if 'ERPM_MAX' in self.constants:
                            self.msg_vesc.data = speed * self.constants['ERPM_MAX']
                        else:
                            self.get_logger().info(
                                'Unable to compute VESC speed because \'ERPM_MAX\' is not set.'
                            )
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
                self.stop()
            elif (
                (direction == Direction.FORWARD and speed > 0)
                or (direction == Direction.BACKWARD and speed < 0)
            ):
                with self.lock:
                    if self.run_mode == RunMode.BASIC:
                        if ('FORWARD_MIN' in self.constants) and ('FORWARD' in self.constants):
                            self.msg.pwm_drive = int(
                                self.constants['FORWARD_MIN'] + abs(speed) * self.constants['FORWARD'])
                        else:
                            self.get_logger().info(
                                'Unable to compute PWM speed because \'FORWARD_MAX\' / \'FORWARD_MIN\' is not set.'
                            )
                            return False

                    elif self.run_mode == RunMode.SIMULATION:
                        self.msg_cmd_vel.linear.x = abs(speed) * self.constants['SIM_SPEEDUP']

                    elif self.run_mode == RunMode.BASIC_VESC:
                        if 'ERPM_MAX' in self.constants:
                            self.msg_vesc.data = abs(speed) * self.constants['ERPM_MAX']
                        else:
                            self.get_logger().info(
                                'Unable to compute VESC speed because \'ERPM_MAX\' is not set.'
                            )
                            return False

            elif (
                (direction == Direction.BACKWARD and speed > 0)
                or (direction == Direction.FORWARD and speed < 0)
            ):
                with self.lock:
                    if self.run_mode == RunMode.BASIC:
                        if ('BACKWARD_MIN' in self.constants) and ('BACKWARD' in self.constants):
                            self.msg.pwm_drive = int(
                                self.constants['BACKWARD_MIN'] + abs(speed) * self.constants['BACKWARD'])
                        else:
                            self.get_logger().info(
                                'Unable to compute PWM speed because \'BACKWARD_MAX\' / \'BACKWARD_MIN\' is not set.'
                            )
                            return False

                    elif self.run_mode == RunMode.SIMULATION:
                        self.msg_cmd_vel.linear.x = -abs(speed) * self.constants['SIM_SPEEDUP']

                    elif self.run_mode == RunMode.BASIC_VESC:
                        if 'ERPM_MAX' in self.constants:
                            self.msg_vesc.data = -abs(speed) * self.constants['ERPM_MAX']
                        else:
                            self.get_logger().info(
                                'Unable to compute VESC speed because \'ERPM_MAX\' is not set.'
                            )
                            return False
            else:
                return False

        elif control_mode == ControlMode.METRIC:
            # Check run modes
            if self.run_mode == RunMode.BASIC:
                self.get_logger().info('Unsupported run mode for data in \'METRIC\' format.')
                return False

            # Check the direction
            if not isinstance(direction, Direction):
                return False

            # Set corresponding variable
            if speed == 0:
                self.stop()
            elif self.run_mode == RunMode.SIMULATION:
                with self.lock:
                    self.msg_cmd_vel.linear.x = speed
            elif 'TO_ERPM' in self.constants:
                with self.lock:
                    self.msg_vesc.data = speed * self.constants['TO_ERPM']
            else:
                self.get_logger().info('Unable to compute VESC speed because \'TO_ERPM\' is not set.')
                return False

        else:
            self.get_logger().info('Unsupported data format.')
            return False

        return True

    def set_forward_speed(self, speed, control_mode=ControlMode.LEGACY):
        """Trampoline function for 'setSpeed()'."""
        return self.set_speed(speed, Direction.FORWARD, control_mode)

    def set_backward_speed(self, speed, control_mode=ControlMode.LEGACY):
        """Trampoline function for 'setSpeed()'."""
        return self.set_speed(speed, Direction.BACKWARD, control_mode)

    def stop(self):
        """Send a calm state speed. It means that the car stops."""

        # Check the self.constants dict
        if not self.constants:
            return False

        with self.lock:
            if 'CALM_SPEED' in self.constants:
                self.msg.pwm_drive = self.constants['CALM_SPEED']

            self.msg_cmd_vel.linear.x = 0
            self.msg_vesc.data = 0

        return True

    def set_steer(self, steer, direction, control_mode=ControlMode.LEGACY):
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

        # Check eStop
        if self.eStop:
            return False

        # Check the self.constants dict
        if not self.constants:
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
            if self.run_mode == RunMode.SIMULATION:
                if steer == 0:
                    self.reset_steer()
                else:
                    self.msg_cmd_vel.angular.z = steer * self.constants['SIM_STEERUP'] \
                                                 * (1.0 if direction == Direction.LEFT else -1.0)

                return True

            # Set corresponding variable
            if direction == Direction.LEFT:
                with self.lock:
                    if ('LEFT_MIN' in self.constants) and ('LEFT' in self.constants):
                        self.msg.pwm_angle = int(self.constants['LEFT_MIN'] + steer * self.constants['LEFT'])
                    else:
                        self.get_logger().info(
                            'Unable to compute PWM steer because \'LEFT_MAX\' / \'LEFT_MIN\' is not set.'
                        )
                        return False

            elif direction == Direction.RIGHT:
                with self.lock:
                    if ('RIGHT_MIN' in self.constants) and ('RIGHT' in self.constants):
                        self.msg.pwm_angle = int(self.constants['RIGHT_MIN'] + steer * self.constants['RIGHT'])
                    else:
                        self.get_logger().info(
                            'Unable to compute PWM steer because \'RIGHT_MAX\' / \'RIGHT_MIN\' is not set.'
                        )
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
            if self.run_mode == RunMode.SIMULATION:
                if steer == 0:
                    self.reset_steer()
                else:
                    self.msg_cmd_vel.angular.z = abs(steer) * self.constants['SIM_STEERUP'] \
                                                 * (1.0 if direction == Direction.LEFT else -1.0)

                return True

            # Set corresponding variable
            if steer == 0:
                self.reset_steer()
            elif (
                (direction == Direction.LEFT and steer > 0)
                or (direction == Direction.RIGHT and steer < 0)
            ):
                with self.lock:
                    if ('LEFT_MIN' in self.constants) and ('LEFT' in self.constants):
                        self.msg.pwm_angle = int(self.constants['LEFT_MIN'] + abs(steer) * self.constants['LEFT'])
                    else:
                        self.get_logger().info(
                            'Unable to compute PWM steer because \'LEFT_MAX\' / \'LEFT_MIN\' is not set.'
                        )
                        return False

            elif (
                (direction == Direction.RIGHT and steer > 0)
                or (direction == Direction.LEFT and steer < 0)
            ):
                with self.lock:
                    if ('RIGHT_MIN' in self.constants) and ('RIGHT' in self.constants):
                        self.msg.pwm_angle = int(self.constants['RIGHT_MIN'] + abs(steer) * self.constants['RIGHT'])
                    else:
                        self.get_logger().info(
                            'Unable to compute PWM steer because \'RIGHT_MAX\' / \'RIGHT_MIN\' is not set.'
                        )
                        return False

            else:
                return False

        elif control_mode == ControlMode.ANGULAR:
            # Check the direction
            if not isinstance(direction, Direction):
                return False

            # Simulation mode
            if self.run_mode == RunMode.SIMULATION:
                if steer == 0:
                    self.reset_steer()
                else:
                    self.msg_cmd_vel.angular.z = steer if direction == Direction.LEFT else -steer

                return True

            # Check the required parameters
            if ('SERVO_LEFT_MAX' not in self.constants) or ('SERVO_RIGHT_MAX' not in self.constants):
                self.get_logger().info(
                    'Unable to compute PWM steer because \'SERVO_LEFT_MAX\' / \'SERVO_RIGHT_MAX\' is not set.'
                )
                return False

            if steer == 0:
                self.reset_steer()
            elif (
                (direction == Direction.LEFT and steer > 0)
                or (direction == Direction.RIGHT and steer < 0)
            ):
                with self.lock:
                    if ('LEFT_MIN' in self.constants) and ('LEFT' in self.constants):
                        self.msg.pwm_angle = int(
                            self.constants['LEFT_MIN'] + min(abs(steer) / self.constants['SERVO_LEFT_MAX'], 1.0) *
                            self.constants[
                                'LEFT'])
                    else:
                        self.get_logger().info(
                            'Unable to compute PWM steer because \'LEFT_MAX\' / \'LEFT_MIN\' is not set.'
                        )
                        return False

            elif (
                (direction == Direction.RIGHT and steer > 0)
                or (direction == Direction.LEFT and steer < 0)
            ):
                with self.lock:
                    if ('RIGHT_MIN' in self.constants) and ('RIGHT' in self.constants):
                        self.msg.pwm_angle = int(
                            self.constants['RIGHT_MIN']
                            + min(abs(steer) / self.constants['SERVO_RIGHT_MAX'], 1.0)
                            * self.constants['RIGHT']
                        )
                    else:
                        self.get_logger().info(
                            'Unable to compute PWM steer because \'RIGHT_MAX\' / \'RIGHT_MIN\' is not set.'
                        )
                        return False

        else:
            self.get_logger().info('Unsupported data format.')
            return False

        return True

    def set_left_steer(self, steer, control_mode=ControlMode.LEGACY):
        """Trampoline function for 'setSteer()'."""
        return self.set_steer(steer, Direction.LEFT, control_mode)

    def set_right_steer(self, steer, control_mode=ControlMode.LEGACY):
        """Trampoline function for 'setSteer()'."""
        return self.set_steer(steer, Direction.RIGHT, control_mode)

    def reset_steer(self):
        """Send a calm state steer. It means that the car turns the wheels to go straight."""

        # Check the self.constants dict
        if not self.constants:
            return False

        with self.lock:
            if 'CALM_STEER' in self.constants:
                self.msg.pwm_angle = self.constants['CALM_STEER']

            self.msg_cmd_vel.angular.z = 0
        pass

    def api_callback(self, data: DriveApiValues):
        """Obtain requested speed/steering from ROS topic.

        Arguments:
        data -- structure received on topic /drive_api/command, defined by drive_api_values
        """

        if data.velocity < 0:
            self.stop()
        elif data.forward:
            self.set_forward_speed(data.velocity)
        else:
            self.set_backward_speed(data.velocity)

        if data.steering < 0:
            self.reset_steer()
        elif data.right:
            self.set_right_steer(data.steering)
        else:
            self.set_left_steer(data.steering)

        pass

    def command_callback(self, data: CommandArrayStamped):
        """Obtain requested speed/steering from ROS topic.

        Arguments:
        data -- structure received on topic /command, defined by CommandArrayStamped
        """

        # Check eStop
        if self.eStop:
            return

        if len(data.commands) > 0:
            for c in data.commands:
                if c.command == 'speed' and len(c.parameters) > 0:
                    for p in c.parameters:
                        success = False

                        # LEGACY
                        if p.parameter == 'forward':
                            if p.value < 0:
                                success = self.stop()
                            else:
                                success = self.set_forward_speed(p.value, ControlMode.LEGACY)
                        elif p.parameter == 'backward':
                            if p.value < 0:
                                success = self.stop()
                            else:
                                success = self.set_backward_speed(p.value, ControlMode.LEGACY)
                        # JOINT
                        elif p.parameter == 'norm' or p.parameter == 'joint':
                            success = self.set_forward_speed(p.value, ControlMode.JOINT)
                        # METRIC
                        elif p.parameter == 'metric' or p.parameter == 'm/s':
                            success = self.set_forward_speed(p.value, ControlMode.METRIC)

                        if not success:
                            self.get_logger().info(
                                f'Unable to process \'{c.command}\' parameter: {p.parameter} ({p.value:f})'
                            )
                            continue

                        break
                elif c.command == 'steer' and len(c.parameters) > 0:
                    for p in c.parameters:
                        success = False

                        # LEGACY
                        if p.parameter == 'left':
                            if p.value < 0:
                                success = self.stop()
                            else:
                                success = self.set_left_steer(p.value, ControlMode.LEGACY)
                        elif p.parameter == 'right':
                            if p.value < 0:
                                success = self.stop()
                            else:
                                success = self.set_right_steer(p.value, ControlMode.LEGACY)
                        # JOINT
                        elif p.parameter == 'norm' or p.parameter == 'joint':
                            success = self.set_left_steer(p.value, ControlMode.JOINT)
                        # ANGULAR
                        elif p.parameter == 'deg':
                            success = self.set_left_steer(math.radians(p.value), ControlMode.ANGULAR)
                        elif p.parameter == 'rad':
                            success = self.set_left_steer(p.value, ControlMode.ANGULAR)

                        if not success:
                            self.get_logger().info(
                                f'Unable to process \'{c.command}\' parameter: {p.parameter} ({p.value:f})'
                            )
                            continue

                        break
                else:
                    self.get_logger().info(f'Unable to process command: \'{c.command}\'')

        pass

    def estop_callback(self, data):
        """Obtain emergency stop message from ROS topic.

        This message can be also used to stop/start simulation.

        Arguments:
        data -- structure received on topic /eStop, defined by std_msgs.msg/Bool
        """

        self.eStop = data.data

        if self.eStop:
            # Set calm states
            self.stop()
            self.reset_steer()

        pass


pass


# note: ROS 2 examples, docs, tutorial and other resources and pretty inconsistent
#       about the style of this function.
#       When main is called from autogenerated executable, no arguments are passed
#       (but sys.argv[0] may be tweaked).
#       When to rclpy.init is passed args=None rclpy.init(args=None), it defaults to
#       sys.argv, see context.py init:
#         `rclpy_implementation.rclpy_init(args if args is not None else sys.argv, capsule)`
#       So instead, we set args to sys.argv ourselves so we can extract what interests us.
def main(args=None):
    if args is None:
        args = sys.argv

    rclpy.init(args=args)

    print(f'main args = {args}')

    # TODO: better solution? best practice?
    simulation = 'simulation=true' in args
    use_vesc = 'use_vesc=true' in args

    node = None

    try:
        node = DriveApiNode(
            simulation=simulation,
            use_vesc=use_vesc,
        )
        rclpy.spin(node)
    except InitError as e:
        print(f'InitError: {e}', file=sys.stderr)
        pass
    except KeyboardInterrupt:
        pass

    if node is not None:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
