#!/usr/bin/env bash

set -e

# usage: save_slam.sh name

# see also https://google-cartographer-ros.readthedocs.io/en/latest/assets_writer.html

if [[ -z $1 ]]; then
	echo "Please enter name as the first argument."
	exit 1
fi

ros2 service call /cartographer/finish_trajectory \
	cartographer_ros_msgs/srv/FinishTrajectory \
	'{trajectory_id: 0}'

ros2 service call /cartographer/write_state \
	cartographer_ros_msgs/srv/WriteState \
	"{filename: \"$1.pbstream\"}"
