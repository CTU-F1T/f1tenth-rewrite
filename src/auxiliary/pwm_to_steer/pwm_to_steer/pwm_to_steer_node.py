import rclpy
from rclpy.node import Node
import math
from std_msgs.msg import Float64
from teensy_drive_msgs.msg import PwmHigh
from typing import List, Tuple, Dict, Any
import sys
from rcl_interfaces.msg import \
    ParameterDescriptor, \
    ParameterType, \
    Parameter, \
    SetParametersResult, \
    FloatingPointRange, \
    IntegerRange


class InitError(Exception):
    pass


class PWMToSteer(Node):
    def __init__(self):
        super().__init__(node_name='pwm_to_steer')

        # configuration
        self.config_initialized: bool = False
        # config map
        # initialized from parameters in add_on_set_parameters_callback
        self.config: Dict[str, Any] = {}
        # first register parameters callbacks
        # (so they are triggered also for the initial invocation when declaring the parameters)
        # note: the order matters!
        #   callbacks are added in front to the list of callbacks
        #   so we need to add them in the reverse order
        self.add_on_set_parameters_callback(self.set_parameters_copy_and_recalculate_callback)
        self.add_on_set_parameters_callback(self.set_parameters_validate_callback)

        self.get_logger().info(f'initializing...')

        # then declare all parameters and let their values automatically get populated with CLI or files overrides)
        self.setup_parameters()
        # set flag to indicate that all parameters are loaded
        # and that set_parameters_copy_and_recalculate_callback should invoke recalculate_derived_configs
        # for any future updates
        self.config_initialized = True
        # manually invoke recalculate_derived_configs for the first time
        self.recalculate_derived_configs()

        self.pwm_sub = self.create_subscription(
            msg_type=PwmHigh,
            topic='/pwm_high',
            callback=self.pwm_callback,
            qos_profile=1,
        )

        self.pub_steer = self.create_publisher(
            msg_type=Float64,
            topic='sensors/servo_position_command',
            qos_profile=1
        )

    def set_parameters_copy_and_recalculate_callback(self, parameters: List[Parameter]) -> SetParametersResult:
        """Called when there is a request to set one or multiple parameters

        Called even for the initial parameter declaration (if registered before the declaration).

        Registered using self.add_on_set_parameters_callback(self.reconfigure_callback).

        Called for each parameter separately (i.e. len(parameters) == 1),
        unless multiple parameters are set using set_parameters_atomically (then len(parameters) >= 1).

        Before this callback is called, parameters' values are validated against their specified constraints (if any).
        If type or constraints validation fails, this callback will not be called at all.

        If this callback returns SetParametersResult(successful=False), the values will not be set.

        """

        # self.get_logger().info('set_parameters_copy_and_recalculate_callback')

        # copy parameters to config
        for param in parameters:
            # print(f'  param={param.name} value={param.value}')
            if param.name == 'angular_steering.left_max' or param.name == 'angular_steering.right_max':
                self.config[param.name] = math.radians(param.value)  # convert degrees to radians
            else:
                self.config[param.name] = param.value  # use the value as it is
            pass

        if self.config_initialized:
            self.recalculate_derived_configs()

        return SetParametersResult(successful=True)


    def set_parameters_validate_callback(self, parameters: List[Parameter]) -> SetParametersResult:
        """Called when there is a request to set one or multiple parameters

        Called even for the initial parameter declaration (if registered before the declaration).

        Registered using self.add_on_set_parameters_callback(self.reconfigure_callback).

        Called for each parameter separately (i.e. len(parameters) == 1),
        unless multiple parameters are set using set_parameters_atomically (then len(parameters) >= 1).

        Before this callback is called, parameters' values are validated against their specified constraints (if any).
        If type or constraints validation fails, this callback will not be called at all.

        If this callback returns SetParametersResult(successful=False), the values will not be set.

        """

        # self.get_logger().info('set_parameters_validate_callback')

        # validate parameters
        for param in parameters:
            # print(f'  param={param.name} value={param.value}')
            if param.value is None:
                return SetParametersResult(
                    successful=False,
                    reason=f'missing value for {param.name}'
                )

        # if we pass successful=False, parameter value will not be set
        # if parameter set was attempted using self.set_parameter*, node will exit with an error
        # if parameter set was attempted remotely, the remote caller is just passed the result
        # (failure and the optional configurable reason='why it was unsuccessful')
        return SetParametersResult(successful=True)

        pass

    def setup_parameters(self):

        # TODO: uninitialized values for statically typed params do not cause failure?

        # (name, value, descriptor)
        params_def: List[Tuple[str, Any, ParameterDescriptor]] = [
            (
                'pwm.steering.calm_value',
                None,
                ParameterDescriptor(
                    type=ParameterType.PARAMETER_INTEGER,
                    description='TODO',
                )
            ),
            (
                'pwm.steering.left.min',
                None,
                ParameterDescriptor(
                    type=ParameterType.PARAMETER_INTEGER,
                    description='TODO',
                )
            ),
            (
                'pwm.steering.left.max',
                None,
                ParameterDescriptor(
                    type=ParameterType.PARAMETER_INTEGER,
                    description='TODO',
                )
            ),
            (
                'pwm.steering.right.min',
                None,
                ParameterDescriptor(
                    type=ParameterType.PARAMETER_INTEGER,
                    description='TODO',
                )
            ),
            (
                'pwm.steering.right.max',
                None,
                ParameterDescriptor(
                    type=ParameterType.PARAMETER_INTEGER,
                    description='TODO',
                )
            ),
            (
                'angular_steering.left_max',
                None,
                ParameterDescriptor(
                    type=ParameterType.PARAMETER_DOUBLE,
                    description='in degrees',
                    floating_point_range=[FloatingPointRange(
                        from_value=0.0,
                        to_value=90.0,
                        step=0.0,
                    )],
                )
            ),
            (
                'angular_steering.right_max',
                None,
                ParameterDescriptor(
                    type=ParameterType.PARAMETER_DOUBLE,
                    description='in degrees',
                    floating_point_range=[FloatingPointRange(
                        from_value=0.0,
                        to_value=90.0,
                        step=0.0,
                    )],
                )
            ),
        ]

        self.declare_parameters(
            namespace='',
            parameters=params_def,
        )

        pass

    def recalculate_derived_configs(self):
        self.get_logger().info('recalculate_derived_configs')
        self.config['pwm.steering.left.range'] = \
            self.config['pwm.steering.left.max'] - self.config['pwm.steering.left.min']
        self.config['pwm.steering.right.range'] = \
            self.config['pwm.steering.right.max'] - self.config['pwm.steering.right.min']
        pass

    def pwm_callback(self, data: PwmHigh):
        pwm_steer = data.period_str

        if pwm_steer > self.config['pwm.steering.right.min']:
            # right

            # self.msg.pwm_angle = int(
            #     self.config['pwm.steering.right.min']
            #     + min(abs(steer) / self.config['angular_steering.right_max'], 1.0)
            #     * self.config['pwm.steering.right.range']
            # )

            steer = -1.0 * (pwm_steer - self.config['pwm.steering.right.min']) / self.config['pwm.steering.right.range'] * self.config['angular_steering.right_max']
            # penis
        elif pwm_steer < self.config['pwm.steering.left.min']:
            # left

            # self.msg.pwm_angle = int(
            #     self.config['pwm.steering.left.min']
            #     + min(abs(steer) / self.config['angular_steering.left_max'], 1.0)
            #     * self.config['pwm.steering.left.range'])

            steer = (pwm_steer - self.config['pwm.steering.left.min']) / self.config['pwm.steering.left.range'] * self.config['angular_steering.left_max']

        else:
            steer = 0.0

        # steer = steer / 180 * math.pi

        steer_msg = Float64()
        steer_msg.data = steer

        self.pub_steer.publish(steer_msg)


def main(args=None):
    if args is None:
        args = sys.argv

    rclpy.init(args=args)

    print(f'main args = {args}')

    try:
        node = PWMToSteer()
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
