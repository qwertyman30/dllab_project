#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <modulation_rl/modulation.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include "tf/transform_datatypes.h"
#include <tf_conversions/tf_eigen.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/RobotState.h>
#include <moveit/robot_state/conversions.h>
//#include <time.h>  
//#include <chrono>
#include <rosbag/bag.h>

class DynamicSystem_observeVel
{
private:
  //! The node handle we'll be using
 
  ros::Publisher traj_visualizer_;
  moveit_msgs::DisplayTrajectory display_trajectory_;
  ros::Publisher gripper_visualizer_;
  
  ros::NodeHandle* nh_;
  
  robot_state::RobotStatePtr kinematic_state_;
  robot_state::JointModelGroup* joint_model_group;

  random_numbers::RandomNumberGenerator rng_;

  tf::Transform currentGripperGOAL_;
  tf::Transform current_base_goal_;
  tf::Transform currentGripperTransform_;
  tf::Transform currentBaseTransform_;

  tf::Transform base_vel_;
  tf::Transform rel_gripper_goal_;
  tf::Transform rel_gripper_pose_;
  tf::Transform rel_base_goal_;

  tf::Transform planned_base_vel_;
  tf::Transform planned_gripper_vel_;
  tf::Transform planned_gripper_vel_rel_;
  tf::Quaternion planned_gripper_dq_;

  int ik_error_count_ = 0;
  int marker_counter_ = 0;
  bool rnd_start_pose_;
  double max_planner_velocity_;
  double min_goal_dist_;
  double max_goal_dist_;

  void set_new_random_goal();
  std::vector<double> build_obs_vector(tf::Transform planned_base_vel, tf::Transform planned_gripper_vel_rel, tf::Quaternion planned_gripper_dq_rel);
  std::tuple<tf::Transform, tf::Transform, tf::Quaternion> plan_new_velocities();
  int calc_done_ret(bool found_ik, int max_allow_ik_errors);
  double calc_reward(bool found_ik, double penalty_scaling, double modulation_alpha, double modulation_lambda1, double modulation_lambda2, double modulation_lambda3);

public:
  DynamicSystem_observeVel(bool rnd_start_pose, uint32_t seed);

  //double simulate_env_step(Eigen::VectorXf& curr_speed);
  std::vector<double> simulate_env_step(int max_allow_ik_errors, double penalty_scaling, double modulation_alpha, double modulation_lambda1, double modulation_lambda2, double modulation_lambda3);
  std::vector< double > reset();
  void visualize_robot_pose(bool vis_now, bool ik_fail, std::string logfile);
   

};