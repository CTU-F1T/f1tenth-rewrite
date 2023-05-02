#!/usr/bin/env bash

set -ex

ros2 topic pub /eStop -1 -w 0 std_msgs/msg/Bool 'data: False'
