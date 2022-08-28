# F1/10 @ CTU (ROS 2 port)

An attempt to port the CTU's **f1tenth** codebase (originally at `git@rtime.felk.cvut.cz:f1tenth`)
from **ROS 1** Kinetic Kame 🐢 to **ROS 2**.

**Current status:** Working Follow the Gap on the real CTU's F1TENTH car (with NVIDIA Jetson TX2, Teensy 3.2, VESC) and
in the **Stage simulator**.

_The work in this repository is a part of my [bachelor's thesis][fel-bachelors-thesis] where more details can be found._


## Content

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Development](#development)
	- [ROS 2 version](#ros-2-version)
	- [Note about OS support](#note-about-os-support)
	- [Other requirements](#other-requirements)
	- [Installation](#installation)
	- [Building](#building)
	- [Running](#running)
	- [Usage with IDEs](#usage-with-ides)
- [Packages overview](#packages-overview)
- [Notes](#notes)
	- [Pure Python packages cannot contain messages definitions](#pure-python-packages-cannot-contain-messages-definitions)
	- [Pure Python packages' assets are not symlinked correctly](#pure-python-packages-assets-are-not-symlinked-correctly)
	- [Python entrypoints `main()` inconsistencies](#python-entrypoints-main-inconsistencies)
	- [Others](#others)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Development


### ROS 2 version

**The main requirement** is a working ROS 2 installation. We tested our code with different ROS 2 versions:
* ROS 2 [Foxy Fitzroy][ros2-foxy] 🦊 _(May 2020)_
* ROS 2 [Galactic Geochelone][ros2-galactic] 🌌 _(May 2021)_
* ROS 2 [Humble Hawksbill][ros2-humble] 💙 – **recommended** 👈 _(May 2022)_
* ROS 2 [Rolling Ridley][ros2-rolling] 🎲 – _As of 08/28/2022 it works, it might break in the future._


### Note about OS support

ROS 2 targets different platforms. The best supported platform (both by ROS core packages and third-party packages)
is **Ubuntu**. Windows and macOS are supported as well but their support is not as good. See [REP 2000][rep-2000] for
more info. **Our recommendations:**
* Use the specific **Ubuntu** release that is targeted by the ROS 2 version you use. For ROS 2 rolling/humble that's
  Ubuntu 22.04, for ROS 2 galactic/foxy it's Ubuntu 20.04.
* On Windows, [WSL 2][wsl-2] with Ubuntu is probably better than native ROS 2 for Windows (because of third-party
  packages).
* On macOS, ROS 2 and all packages must be built from sources. The best way to manage it is to use 3 different
  workspaces stacked on top of each other:
	1. This repository as the top level workspace.
	2. One underlying workspace with any dependencies (packages) (normally, on Ubuntu, one would install already-built
	   binary packages using `apt`/`rosdep`).
	3. The bottom underlay – workspace with ROS 2 core packages that make up the standard ROS 2. You can use
	   the [pokusew/ros2-build] to automate creating and building the second and the third workspace.

  Note: We successfully tested this project on macOS 10.14.6 with ROS 2 foxy and galactic.


### Other requirements

* [vcstool]
	* Test if you have it using: `vcs --version` (should print `vcs 0.3.0`)
	* You can install it using `sudo apt install python3-vcstool` or `python3 -m pip install -U vcstool`.
* [colcon]
	* Test if you have it (and which extensions) using: `colcon version-check`
	* Usually installed as a part of ROS 2 distribution, otherwise it can be [installed manually][colcon-installation].
	* 👉 [**Follow the instructions here**][ros2-tips-colcon] to install colcon _mixins_ and configure _shell
	  autocompletion_.


### Installation

Make sure you have a [working ROS 2 installation][#ros-2-version]. You can follow the official ROS 2 installation guide,
the one for [**ROS 2 Humble on Ubuntu** (using apt packages) is here][ros2-humble-install-ubuntu-packages].

Then, make sure you have `vcstool` and `colcon` as well, see [Other requirements][#other-requirements].

📌 **Tip:** See the [**ROS 2 Tips & Tricks**][ros2-tips-colcon] to learn useful tips about using ROS 2, `colcon`, and
many more.

Clone this repository and checkout to its root directory:
```bash
git clone git@github.com:pokusew/f1tenth-rewrite.git
cd f1tenth-rewrite
```

Open at least two (clean) terminals, one for building the workspace and the other one(s) for sourcing the built
workspace and running the nodes. Remember
to [always use a different terminal for building the workspace][ros2-tips-colcon] than for sourcing it and running it.


### Building

**Build terminal** (in the project root directory):
```bash
# Source ROS 2 (if you don't do it automatically in your .bashrc, which I do NOT recommend).
# Note: Change the path to your ROS 2 distro accordingly.
source /opt/ros/humble/setup.bash
# Clone external packages' sources using vcstool.
vcs import --input stack.repos --force
# Install any required dependencies using rosdep.
# We have to explicitly ignore
# - Stage because rosdep cannot detect pure CMake packages.
# - slam_toolbox and cartographer_ros because they are not needed for the FTG app
#   and might not be available on all platforms for an easy install using apt.
rosdep install -i --from-paths src -y --skip-keys="Stage slam_toolbox cartographer_ros"
# Install Stage's dependencies manually because Stage is not a ROS package (no package.xml so rosdep cannot work).
# Stage also requires libpng-dev and libjpeg-dev but those are normally already installed.
sudo apt install libfltk1.1-dev libglu1-mesa-dev
# Build the workspace.
# Note:
#   For --mixin flag to work, you need colcon-mixin extension.
#   See https://github.com/pokusew/ros-setup/blob/main/ROS2-Tips.md#colcon for installation instructions.
#   Alternatively, you can use the variant without --mixin flag below.
colcon build --symlink-install --mixin compile-commands
colcon build --symlink-install --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --no-warn-unused-cli
```


### Running

To see that everything is working, we can run the Follow the Gap in the Stage simulator.

**Run terminal 1** (in the project root directory):
```bash
# Source the workspace.
# Note:
#   Any underlay(s) (typically ROS 2 base workspace),
#   that were sourced when this workspace was built, will be sourced as well.
source install/setup.bash
# Run the Follow the Gap in the Stage simulator.
ros2 launch auto stage_sample_ftg.launch.py
```

**Run terminal 2** (in the project root directory):
```bash
# Source the workspace.
source install/setup.bash
# Start the auto (i.e., publishes one bool message containing `False` to the /eStop topic).
auto start
# Auto CLI is automatically added to the PATH when sourcing the workspace.
# Another possibility is to write manually:
ros2 topic pub /eStop -1 std_msgs/msg/Bool 'data: False'
```


### Usage with IDEs

Please refer to the 👉 [Using VSCode and JetBrains IDEs (CLion, PyCharm) with ROS][ros-setup-ide] 👈.


## Packages overview

_Note 1:_ **Bold names** denotes ROS 2 packages. Italic text in parentheses specifies build system, either _(Ament
CMake)_ or _(Pure Python)_ or _(Pure CMake)_.

_Note 2:_ There might _still_ be a few things in the code that need to be explained, clarified, ... Look for **TODO**
comments.

_src/:_
* **auto** _(Ament CMake)_ – Global launch files (ported from the _launchers_ package), configs, and a simple Auto CLI.
* _decision_and_control/_
	* **follow_the_gap_v0** _(Ament CMake)_ – The algorithm implementation and a ROS 2 wrapper node, both in **C++**.
	* **follow_the_gap_v0_ride** _(Pure Python)_ – A node that converts the heading angle from FTG to control commands.
	  Extracted from follow_the_gap_v0 package.
* _messages/_
	* **command_msgs** _(Ament CMake)_ – IDL
	* **drive_api_msgs** _(Ament CMake)_ – IDL
	* **obstacle_msgs** _(Ament CMake)_ – IDL
	* **plan_msgs** _(Ament CMake)_ – IDL
	* **teensy_drive_msgs** _(Ament CMake)_ – IDL
* _perception/_
	* _recognition/_
		* **obstacle_substitution** _(Pure Python)_ – A simple Python node that converts LiDAR scans to obstacles. Each
		  LiDAR point to a CircleObstacle.
		* **cartographer_slam** _(Ament CMake)_ – Configuration and launch files for Google Cartographer.
		* **slam_toolbox_slam** _(Ament CMake)_ – Configuration and launch files for slam_toolbox.
	* _sensors/_
		* **ros2_razor_imu** _(Pure Python)_ – ROS 2 driver for 9DoF Razor IMU
		  M0 ([external](https://github.com/pokusew/ros2_razor_imu), currently not cloned,
		  see [stack.repos](./stack.repos)).
* _simulation/_
	* **stage** _(Pure CMake)_ – Stage simulator (C++) ([external](https://github.com/rtv/Stage)).
	* **stage_ros2** _(Ament CMake)_ – ROS 2 bindings (C++) for the Stage
	  simulator ([external](https://github.com/pokusew/stage_ros2)).
* **storage** _(Ament CMake)_ – Data files.
* _vehicle_platform/_
	* **drive_api** – Drive-API Python node for controlling the vehicle.
	* **teensy_drive** – _(Ament CMake)_ A C++ node that handles communication between ROS 2 and the Teensy 3.2 MCU
	  which is responsible for servo control, RC control and emergency manual override.
	* _vesc/_ – VESC ROS 2 driver and related packages ([external](https://github.com/pokusew/vesc/tree/ros2-pokusew)).


## Notes


### Pure Python packages cannot contain messages definitions

ROS 2 _Pure Python_ packages cannot contain IDL definitions (messages, services)
_(at least for now, it may be supported in the future)_. It is possible create _Ament CMake_ packages with Python code.
But the best practice is to split the messages definitions into separate packages (with `_msgs` as a suffix,
per-convention). It speeds up the build and provides greater flexibility.


### Pure Python packages' assets are not symlinked correctly

`colcon --symlink-install` does **not** work correctly with Pure Python packages. Launch files, config files and any
other asset files are not symlinked.
* see [colcon/colcon-core#407](https://github.com/colcon/colcon-core/issues/407)
* see [ros2/launch#187](https://github.com/ros2/launch/issues/187)

The best **workarounds**:
1. Prefer putting launch files and config files to _Ament CMake_ packages which are symlinked correctly.
2. “Put implementations of the launch methods in the package modules (which indirectly get symlink installed) and simply
   making a one line call out to it from the launch script. Since the one line call out rarely changes, it just needs
   the one install (build).” This
   idea [comes from this comment](https://github.com/colcon/colcon-core/issues/169#issuecomment-531517276).
2. Accept the situation and build after every change to the assets file of a Pure Python package. Speed up the build by
   selecting only that package using `colcon build --package-select <package>`.


### Python entrypoints `main()` inconsistencies

ROS 2 examples, docs, tutorial and other resources and pretty inconsistent about the style of the `main()` function.

When `main()` is called from autogenerated executable, **no arguments** are passed
(but `sys.argv[0]` may be tweaked).

When `rclpy.init(args=None)` is called, the `args` defaults to
`sys.argv`, see `rclpy/context.py`, **line 59**:
```python
# rclpy/context.py, line 59:
rclpy_implementation.rclpy_init(args if args is not None else sys.argv, capsule)
```

So instead, we might as well set `args` to `sys.argv` directly ourselves, so we can extract what interests us.

So the final `main()` we use for Python entrypoints is:
```python
import sys
import rclpy


def main(args=None):
	if args is None:
		args = sys.argv

		rclpy.init(args=args)

# ... further code
```


### Others

* **There is no "global parameter server" in ROS 2**
	* see https://discourse.ros.org/t/ros2-global-parameter-server-status/10114
	* see https://github.com/fujitatomoya/ros2_persist_parameter_server


<!-- links references -->

[fel-bachelors-thesis]: https://github.com/pokusew/fel-bachelors-thesis

[ros2-foxy]: https://docs.ros.org/en/foxy/index.html

[ros2-galactic]: https://docs.ros.org/en/galactic/index.html

[ros2-humble]: https://docs.ros.org/en/humble/index.html

[ros2-rolling]: https://docs.ros.org/en/rolling/index.html

[ros2-humble-install-ubuntu-packages]: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html

[rep-2000]: https://ros.org/reps/rep-2000.html

[wsl-2]: https://docs.microsoft.com/en-us/windows/wsl/compare-versions#whats-new-in-wsl-2

[pokusew/ros2-build]: https://github.com/pokusew/ros2-build

[vcstool]: https://github.com/dirk-thomas/vcstool

[colcon]: https://colcon.readthedocs.io/en/released/index.html

[colcon-installation]: https://colcon.readthedocs.io/en/released/user/installation.html

[colcon-mixin]: https://colcon.readthedocs.io/en/released/reference/verb/mixin.html

[ros2-tips-colcon]: https://github.com/pokusew/ros-setup/blob/main/ROS-2-Tips-Tricks.md#colcon

[ros-setup-ide]: https://github.com/pokusew/ros-setup/blob/main/ide/README.md
