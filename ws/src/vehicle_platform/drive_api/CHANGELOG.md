# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/).

## Unreleased
## 0.3.2 - 2021-01-11
### Fixed
- Compute 'ERPM_MAX' with proper equation.

## 0.3.1 - 2020-09-21
### Fixed
- PWM constants are not computed in `SIMULATION` mode (and `cmd_vel` are not computed in `BASIC` mode), therefore, it is able to run in `SIMULATION` mode without car parameters.
- Simulation action modifiers are now properly received from the Parameter Server.

## 0.3.0 - 2020-09-11
### Added
- Added support for direct specification of VESC related constants by setting `ERPM_MAX` and `TO_ERPM` variables.
- Added `ANGULAR` control mode, which allows to control the steering angle in `rad` or `deg`.
- Added `METRIC` support for `SIMULATION` mode.
- Added `~rate` parameter for external setting of publish rate of the messages.
- Added `~speed_modifier` and `~steer_modifier` to increase the action while using `LEGACY` or `JOINT` control modes in `SIMULATION` mode.

### Changed
- Legacy message type `drive_api_values` is no longer required for running the node.
- Simulation publishing topic `/cmd_vel` is created only in this mode.
- PWM constants are no longer required for running the node in `SIMULATION` mode.
- Improved error messages while handing `/command` messages to be more verbose.
- Ordinary publishing topic `/drive_pwm` is created only in non-`SIMULATION` mode.

## 0.2.0 - 2019-12-17
### Added
- Added new enum `ControlMode` that can be used to select a method of controlling the vehicle. Original control mode is named `LEGACY`.
- Added `JOINT` control mode, which is specified only by two numbers within <-1, 1> interval.
- Added `METRIC` control mode, which allows to control the speed in `m.s^-1`.
- Added `command_msgs/Command`, `command_msgs/CommandParameter` and `command_msgs/CommandArrayStamped` message types (not required to run the node).
- Added `command_callback` function that handles `command_msgs/CommandArrayStamped` messages that can also contain control mode.
- Added VESC support (`LEGACY` and `JOINT` control modes are mapped on max rotation of the motor, computed from parameters).
- Added new enum `RunMode` that is set on node's start and has three options: `BASIC`, `BASIC_VESC` (speed controlled by VESC, steering by Teensy) and `SIMULATION`.

## 0.1.5 - 2019-05-31
### Changed
- Added queue size for Subscribers to avoid stacking of old messages.

## 0.1.4 - 2019-05-08
### Fixed
- Steering direction for `/cmd_vel`.

## 0.1.3 - 2019-05-02
### Added
- Added simulation mode that does not require internal car parameters.
- Added option to launch the node as anonymous.
- Added `/eStop` Subscriber to avoid sending messages when manual mode is active.

### Fixed
- Added handling of uncaught exception when ROS time jumps.

## 0.1.2 - 2019-03-17
### Added
- Publishing received commands to standardized `/cmd_vel` topic.

## 0.1.1 - 2019-02-11
### Changed
- Updated message `drive_api_values` to allow stopping car and resetting steer.

## 0.1.0 - 2019-02-11
### Added
- Access to the API via ROS topic.
- Added custom message type `drive_api_values`.

## 0.0.1 - 2018-05-29
### Added
- First version of car control API layer (Python module).
