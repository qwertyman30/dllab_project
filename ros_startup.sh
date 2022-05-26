#!/usr/bin/env bash
source /opt/ros/melodic/setup.bash
source devel/setup.bash

echo "Starting roscore"
(roscore &) &> /dev/null
sleep 5
echo "Starting pr2_gazebo"
(roslaunch pr2_gazebo pr2_empty_world.launch &) &> /dev/null
sleep 5
echo "Starting roslaunch"
(roslaunch pr2_moveit_config move_group.launch &) &> /dev/null