#!/usr/bin/env bash
bash ros_startup.sh

source /opt/ros/melodic/setup.bash
source devel/setup.bash

cd src/dllab_modulation_rl
echo "Starting command ${1}"
conda run -n base $1