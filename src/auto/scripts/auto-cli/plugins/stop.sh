#!/usr/bin/env bash

set -ex

ros2 topic pub /eStop -1 --keep-alive 3 std_msgs/msg/Bool 'data: True'
