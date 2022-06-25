#!/usr/bin/env bash

git apply patches/Fix_stage_CMakeLists_txt_Fix_stage_ros2_CMakeLists_txt.patch
# TODO: include the changes in to the patch file
cp patches/stage_ros2_CMakeLists.txt src/simulation/stage_ros2/CMakeLists.txt
