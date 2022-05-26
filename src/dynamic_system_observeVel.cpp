#include <modulation_rl/dynamic_system_observeVel.h>

DynamicSystem_observeVel::DynamicSystem_observeVel(bool rnd_start_pose, uint32_t seed)
    : rng_ { seed }
{
    // initialize rosnode to publish robot visualization msgs
    char** ca;
    int argc=0;
    ros::init(argc,NULL,  "ds");
    this->nh_ = new ros::NodeHandle("TD3_PR2_ik");
    traj_visualizer_ = nh_->advertise<moveit_msgs::DisplayTrajectory>("traj_visualizer", 1);
    gripper_visualizer_ = nh_->advertise<visualization_msgs::Marker>("gripper_goal_visualizer", 1);

    // Load Robot config from moveit movegroup (must be running)
    robot_model_loader::RobotModelLoader robot_model_loader("robot_description");
    robot_model::RobotModelPtr kinematic_model = robot_model_loader.getModel();
    kinematic_state_.reset(new robot_state::RobotState(kinematic_model));
    kinematic_state_->setToDefaultValues();
    joint_model_group = kinematic_model->getJointModelGroup("right_arm");

    // other env parameters we might want to vary at some point.
    max_planner_velocity_ = 0.01;
    min_goal_dist_ = 1.0;
    max_goal_dist_ = 5.0;
    // Initialize poses for robot base and gripper and set new random goal pose for gripper and base
    rnd_start_pose_ = rnd_start_pose;
    std::vector<double> reset_pose = reset();

    // Set startstate for trajectory visualization
    display_trajectory_.model_id = "pr2";
    moveit_msgs::RobotState start_state;
    const std::vector<std::string> joint_names = joint_model_group->getVariableNames();
    int m_num_joints = joint_names.size();//8;
    start_state.joint_state.name.resize(m_num_joints);
    start_state.joint_state.position.resize(m_num_joints);
    start_state.joint_state.velocity.resize(m_num_joints);
    
    std::vector< double > joint_values;
    kinematic_state_->copyJointGroupPositions(joint_model_group, joint_values);
    for (int j = 0 ; j < joint_names.size(); j++)
    {
        start_state.joint_state.name[j] = joint_names[j];
        //std::cout << joint_names[j] <<std::endl;
        start_state.joint_state.position[j] = joint_values[j];
    }
    start_state.multi_dof_joint_state.header.frame_id = "odom_combined";
    start_state.multi_dof_joint_state.joint_names.push_back("world_joint");
    geometry_msgs::Transform startTransform;
    startTransform.translation.x = 0;
    startTransform.translation.y = 0;
    startTransform.translation.z = 0;
    startTransform.rotation.x = 0;
    startTransform.rotation.y = 0;
    startTransform.rotation.z = 0;
    startTransform.rotation.w = 1;
    start_state.multi_dof_joint_state.transforms.push_back(startTransform);
    display_trajectory_.trajectory_start = start_state;
}

void DynamicSystem_observeVel::set_new_random_goal(){
    // Seed random generator
    // std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
    // srand (ms.count());

    // Set new random Goal for the Gripper
    //currentGripperGOAL_.setIdentity();
    //int RandGoalDist = (rand() % 410 + 90);
    //int RandGoalDir = (rand() % 314);
    //int RandGoalZ =  (rand() % 120 + 30); // 70;//
    //int yGoalSign = random() & 1 ? 1 : -1;
    //double xGoal = ((double)RandGoalDist)/100.0*cos(((double)RandGoalDir)/100.0);
    //double yGoal = ((double) yGoalSign)*((double)RandGoalDist)/100.0*sin(((double)RandGoalDir)/100.0);
    //currentGripperGOAL_.setOrigin(tf::Vector3(xGoal,yGoal,((double) RandGoalZ)/100.0) );
    //int RandEuler1 = (rand() % 629);
    //int RandEuler2 = (rand() % 629);
    //int RandEuler3 = (rand() % 629);
    //tf::Quaternion q;
    //q.setRPY(((double)RandEuler1)/100.0,((double)RandEuler2)/100.0,((double)RandEuler3)/100.0);
    //currentGripperGOAL_.setRotation(q);

    // Set new random Goal for the robot base
    current_base_goal_.setIdentity();
    double dist_gripper_base = rng_.uniformReal(min_goal_dist_, max_goal_dist_);
    double rand_base_orientation = rng_.uniformReal(0.0, 3.14);
    int rand_sign = (rng_.uniformInteger(0, 1) == 1) ? 1 : -1;

    double x_base_goal = dist_gripper_base * cos(rand_base_orientation);
    double y_base_goal = ((double) rand_sign) * dist_gripper_base * sin(rand_base_orientation);
    tf::Vector3 vec_to_base(x_base_goal, y_base_goal, 0.0);
    current_base_goal_.setOrigin(tf::Vector3(x_base_goal, y_base_goal, 0.0));

    double base_yaw = rng_.uniformReal(0.0, 6.29);
    tf::Quaternion q_base;
    q_base.setRPY(0.0, 0.0, base_yaw);
    current_base_goal_.setRotation(q_base);

    // set random goal for gripper relative to base
    kinematic_state_->setToRandomPositions(joint_model_group, rng_);
    const Eigen::Affine3d& end_effector_state = kinematic_state_->getGlobalLinkTransform("r_wrist_roll_link");
    tf::transformEigenToTF (end_effector_state, currentGripperGOAL_);
    // transform gripperGoal to World
    currentGripperGOAL_ = current_base_goal_ * currentGripperGOAL_;
}

std::vector< double > DynamicSystem_observeVel::reset(){
    // Set new random goals for base and gripper
    set_new_random_goal();

    // Reset Base to origin
    currentBaseTransform_.setIdentity();
    // Reset relative velocity between gripper and base to zero
//    gripper_vel_rel_.setIdentity();
    base_vel_.setIdentity();

    // Reset Gripper pose to start
    if (rnd_start_pose_){
        // c) RANDOM pose relative to base
        kinematic_state_->setToRandomPositions(joint_model_group, rng_);
        const Eigen::Affine3d& end_effector_state = kinematic_state_->getGlobalLinkTransform("r_wrist_roll_link");
        tf::transformEigenToTF (end_effector_state, currentGripperTransform_);
        // multiplication theoretically unneccessary as long as currentBaseTransform_ is the identity
        rel_gripper_pose_ = currentBaseTransform_.inverse() * currentGripperTransform_;
    } else {
        // a) FIXED OFFSET
        currentGripperTransform_.setIdentity();
        tf::Transform offsetGr;
        offsetGr.setIdentity();
        tf::Vector3 translationOffG(0.45,0.0,0.7);
        offsetGr.setOrigin(translationOffG);
        currentGripperTransform_ = currentGripperTransform_*offsetGr;
        rel_gripper_pose_ = currentBaseTransform_.inverse() * currentGripperTransform_;
        // b) IDENTITY -> IK NEVER FOUND
        // rel_gripper_pose_.setIdentity();
        // currentGripperTransform_.setIdentity();
        // rel_gripper_pose_ = currentBaseTransform_.inverse() * currentGripperTransform_;
    }

    // set relative goals
    rel_gripper_goal_ = currentBaseTransform_.inverse() * currentGripperGOAL_;
    rel_base_goal_ = currentBaseTransform_.inverse() * current_base_goal_;

    // Clear the visualization trajectory
    display_trajectory_.trajectory.clear();
    ik_error_count_ = 0;

    // VISUALISE START GRIPPER POSE -> requires to actually set the pose
    Eigen::Isometry3d state;
    tf::poseTFToEigen(rel_gripper_pose_, state);
    const Eigen::Isometry3d &desiredState = state;
    bool found_ik = kinematic_state_->setFromIK(joint_model_group, desiredState);
    visualize_robot_pose(false, false, "");

    // plan velocities to be modulated and set in next step
    auto [Aplanned_base_vel_, Aplanned_gripper_vel_, Aplanned_gripper_dq_] = plan_new_velocities();
    planned_base_vel_ = Aplanned_base_vel_;
    planned_gripper_vel_ = Aplanned_gripper_vel_;
    planned_gripper_dq_ = Aplanned_gripper_dq_;

    // make relative to base transform
    tf::Transform base_no_trans = currentBaseTransform_;
    base_no_trans.setOrigin(tf::Vector3(0.0,0.0,0.0));
    planned_base_vel_ = base_no_trans.inverse() * planned_base_vel_;
    planned_base_vel_.setRotation(tf::Quaternion(0.0,0.0,0.0,1.0));
    planned_gripper_vel_rel_ = base_no_trans.inverse() * planned_gripper_vel_;
    planned_gripper_vel_rel_.setRotation(tf::Quaternion(0.0,0.0,0.0,1.0));
    tf::Quaternion planned_gripper_dq_rel = (currentGripperTransform_.getRotation() + planned_gripper_dq_) - currentBaseTransform_.getRotation();

    // return observation vector
    return build_obs_vector(planned_base_vel_, planned_gripper_vel_rel_, planned_gripper_dq_rel);
}

std::vector<double> DynamicSystem_observeVel::build_obs_vector(tf::Transform planned_base_vel, tf::Transform planned_gripper_vel_rel, tf::Quaternion planned_gripper_dq_rel){
    // Build the observation vector
    std::vector<double> obs_vector;
    // Consists of the gripper state relative to the base
    obs_vector.push_back(rel_gripper_pose_.getOrigin().x());
    obs_vector.push_back(rel_gripper_pose_.getOrigin().y());
    obs_vector.push_back(rel_gripper_pose_.getOrigin().z());
    obs_vector.push_back(rel_gripper_pose_.getRotation().x());
    obs_vector.push_back(rel_gripper_pose_.getRotation().y());
    obs_vector.push_back(rel_gripper_pose_.getRotation().z());
    obs_vector.push_back(rel_gripper_pose_.getRotation().w());
    // current joint positions (8 values)
    std::vector< double > joint_values;
    kinematic_state_->copyJointGroupPositions(joint_model_group, joint_values);
    for (int j = 0 ; j < joint_values.size(); j++)
    {
        obs_vector.push_back(joint_values[j]);
//        std::cout << joint_names[j] << joint_values[j] << std::endl;
    }
    // next relative velocity between gripper and base agent has to modulate
    // gripper_vel_rel_ is the velocity relative to the coordinates of the base, but not relative to the speed of the base
    obs_vector.push_back(planned_base_vel.getOrigin().x() - planned_gripper_vel_rel.getOrigin().x());
    obs_vector.push_back(planned_base_vel.getOrigin().y() - planned_gripper_vel_rel.getOrigin().y());
    obs_vector.push_back(planned_base_vel.getOrigin().z() - planned_gripper_vel_rel.getOrigin().z());
    // planned change in rotation
    obs_vector.push_back(planned_gripper_dq_rel.x());
    obs_vector.push_back(planned_gripper_dq_rel.y());
    obs_vector.push_back(planned_gripper_dq_rel.z());
    obs_vector.push_back(planned_gripper_dq_rel.w());
    // relative position of the gripper goal
    obs_vector.push_back(rel_gripper_goal_.getOrigin().x());
    obs_vector.push_back(rel_gripper_goal_.getOrigin().y());
    obs_vector.push_back(rel_gripper_goal_.getOrigin().z());
    obs_vector.push_back(rel_gripper_goal_.getRotation().x());
    obs_vector.push_back(rel_gripper_goal_.getRotation().y());
    obs_vector.push_back(rel_gripper_goal_.getRotation().z());
    obs_vector.push_back(rel_gripper_goal_.getRotation().w());
    // relative position of the base goal
    obs_vector.push_back(rel_base_goal_.getOrigin().x());
    obs_vector.push_back(rel_base_goal_.getOrigin().y());
//    obs_vector.push_back(rel_base_goal_.getOrigin().z());
//    obs_vector.push_back(rel_base_goal_.getRotation().x());
//    obs_vector.push_back(rel_base_goal_.getRotation().y());
//    obs_vector.push_back(rel_base_goal_.getRotation().z());
//    obs_vector.push_back(rel_base_goal_.getRotation().w());

    return obs_vector;
}

double DynamicSystem_observeVel::calc_reward(bool found_ik, double penalty_scaling, double modulation_alpha, double modulation_lambda1, double modulation_lambda2, double modulation_lambda3){
    double reward;
//    if (found_ik){
//        reward = 5000.0-(100.0*pow(1.0-modulation_lambda1,2))-(30000.0*pow(modulation_lambda3,2));
//        //reward = 5000.0-(100.0*pow(1.0-modulation_lambda1,2))-(100000.0*pow(modulation_lambda3,2));
//        //reward = 0.1-(0.0001 * pow(1.0-modulation_lambda1,2))-(0.003*pow(modulation_lambda3,2));
//        //ik_error_count_ = 0;
//    }
//    else{
//        reward = -30000;
//        ik_error_count_++;
//    }

    if (found_ik){
//        reward = 1.0-(0.5*pow(1.0-modulation_lambda1,2))-(25*pow(modulation_lambda3,2));
        // multipliers so that each penality roughly has a max of 1
        // modulation_lambda1 in [-2, 2], modulation_lambda2 in [-0.2, 0.2]
        double lambda1_penality = 0.5*pow(1.0-modulation_lambda1,2);
        double lambda3_penalty = 25*pow(modulation_lambda3,2);
        reward = -penalty_scaling*(lambda1_penality + lambda3_penalty);
    } else{
        reward = -1.0;
        ik_error_count_++;
    }
    return reward;
}

int DynamicSystem_observeVel::calc_done_ret(bool found_ik, int max_allow_ik_errors){
    tf::Vector3 vec_to_goal = currentGripperGOAL_.getOrigin() - currentGripperTransform_.getOrigin();
    double dist_to_goal = pow(pow(vec_to_goal.x(),2.0) + pow(vec_to_goal.y(),2.0) + pow(vec_to_goal.z(),2.0), 0.5);

    if (found_ik && dist_to_goal > 0.1)
        return 0;
    else if(found_ik){ // end episode because goal was reached
        return 1;
        //std::cout << " Current Gripper z: "<<  currentGripperTransform_.getOrigin().z() << " Gripper z Goal: " <<  currentGripperGOAL_.getOrigin().z() << std::endl;
    }
    else if(ik_error_count_ > max_allow_ik_errors || dist_to_goal < 0.1)// end episode because limit of failures is reached or goal reached with failure
        return 2;
    else
        return 0;
}

// pure planning, no sideeffects
std::tuple<tf::Transform, tf::Transform, tf::Quaternion> DynamicSystem_observeVel::plan_new_velocities(){
    // Set new Gripper velocity based on distance to current gripper Goal
    tf::Vector3 vec_to_goal = currentGripperGOAL_.getOrigin() - currentGripperTransform_.getOrigin();
    tf::Transform planned_gripper_vel;
    planned_gripper_vel.setIdentity();
    planned_gripper_vel.setOrigin(vec_to_goal / 100.0);
    tf::Quaternion planned_gripper_dq = (currentGripperGOAL_.getRotation() - currentGripperTransform_.getRotation()) / 100.0;
    // get max of (x, y, z) velocities. If any of these > max_planner_velocity_: scale all of them down so that the largest is max_planner_velocity_
    double max_vel_comp = std::max(planned_gripper_vel.getOrigin().x(), planned_gripper_vel.getOrigin().y());
    max_vel_comp = std::max(max_vel_comp, planned_gripper_vel.getOrigin().z());
    if (max_vel_comp > max_planner_velocity_){
        planned_gripper_vel.setOrigin(tf::Vector3 (planned_gripper_vel.getOrigin().x()*max_planner_velocity_/max_vel_comp,
                                                    planned_gripper_vel.getOrigin().y()*max_planner_velocity_/max_vel_comp,
                                                    planned_gripper_vel.getOrigin().z()*max_planner_velocity_/max_vel_comp));
    }

    // calculate new unmodulated base velocity based on current base goal
    tf::Vector3 vec_to_base_goal = current_base_goal_.getOrigin() - currentBaseTransform_.getOrigin();
    tf::Transform planned_base_vel;
    planned_base_vel.setIdentity();
    planned_base_vel.setOrigin(vec_to_base_goal / 100.0);
    max_vel_comp = std::max(planned_base_vel.getOrigin().x(), planned_base_vel.getOrigin().y());
    if (max_vel_comp > max_planner_velocity_){
        planned_base_vel.setOrigin(tf::Vector3 (planned_base_vel.getOrigin().x()*max_planner_velocity_/max_vel_comp,
                                        planned_base_vel.getOrigin().y()*max_planner_velocity_/max_vel_comp,
                                        0.0));
    }

    return {planned_base_vel, planned_gripper_vel, planned_gripper_dq};
}

std::vector<double> DynamicSystem_observeVel::simulate_env_step(int max_allow_ik_errors, double penalty_scaling, double modulation_alpha, double modulation_lambda1, double modulation_lambda2, double modulation_lambda3){
//std::vector<double> DynamicSystem_observeVel::simulate_env_step(double modulation_alpha, double modulation_lambda1, double modulation_lambda3){
    modulation_lambda2 = 1.0;

    // set new gripper pose with planned velocity and rotation
    currentGripperTransform_.setOrigin(currentGripperTransform_.getOrigin() + planned_gripper_vel_.getOrigin());
    currentGripperTransform_.setRotation(currentGripperTransform_.getRotation() + planned_gripper_dq_);  // addition not problematic for small dq

    // Modulate planned base velocity and set it:
    // i) calculate new (modulated) relative speed of the base to the gripper
    Eigen::Vector2f relative_gripper_base_speed;
    relative_gripper_base_speed << planned_base_vel_.getOrigin().x() - planned_gripper_vel_rel_.getOrigin().x(), planned_base_vel_.getOrigin().y() - planned_gripper_vel_rel_.getOrigin().y();
    //std::cout << "gripper_vel_rel_: (" << gripper_vel_rel_.getOrigin().x() << ", " << gripper_vel_rel_.getOrigin().y() << ")" << std::endl;
    //std::cout << "gripper_vel: (" << gripper_vel.getOrigin().x() << ", " << gripper_vel.getOrigin().y() << ", " << gripper_vel.getOrigin().z() << ")" << std::endl;
    modulation::compModulation(modulation_alpha, modulation_lambda1, modulation_lambda2, relative_gripper_base_speed);
    //std::cout << "Base velocity after modulation: (" << relative_gripper_base_speed(0) << ", " << relative_gripper_base_speed(1) << ")" << std::endl;

    // ii) based on that calculate corresponding new base speed
    tf::Vector3 translationVelBase(relative_gripper_base_speed(0), relative_gripper_base_speed(1), 0.0);
    base_vel_.setOrigin(translationVelBase + planned_gripper_vel_rel_.getOrigin());
    base_vel_.setOrigin(tf::Vector3(base_vel_.getOrigin().x(), base_vel_.getOrigin().y(), 0.0));
    tf::Transform base_no_trans = currentBaseTransform_;
    base_no_trans.setOrigin(tf::Vector3(0.0,0.0,0.0));
    base_vel_ = base_no_trans * base_vel_;

    // iii) set the new base pose
    currentBaseTransform_.setOrigin(currentBaseTransform_.getOrigin() + base_vel_.getOrigin());
    tf::Quaternion Q3(tf::Vector3(0.0,0.0,1.0), modulation_lambda3);
    currentBaseTransform_.setRotation(Q3 * currentBaseTransform_.getRotation());

    // Update relative position of the base and gripper_goal to the base
    rel_gripper_goal_ = currentBaseTransform_.inverse() * currentGripperGOAL_;
    rel_base_goal_ = currentBaseTransform_.inverse() * current_base_goal_;

    // Update rel_gripper_pose_, perform IK checks
    Eigen::Isometry3d state;
    rel_gripper_pose_ = currentBaseTransform_.inverse() * currentGripperTransform_;
    tf::poseTFToEigen(rel_gripper_pose_, state);
    const Eigen::Isometry3d &desiredState = state;
    bool found_ik = kinematic_state_->setFromIK(joint_model_group, desiredState);

    // reward and check if episode has finished -> Distance gripper to goal
    double reward = calc_reward(found_ik, penalty_scaling, modulation_alpha, modulation_lambda1, modulation_lambda2, modulation_lambda3);
    int done_ret = calc_done_ret(found_ik, max_allow_ik_errors);

    // Add the current robot state to the visualization trajectory (not actually visualizing)
    visualize_robot_pose(false, !found_ik, "");

    // plan velocities to be modulated and set in next step
    auto [Aplanned_base_vel_, Aplanned_gripper_vel_, Aplanned_gripper_dq_] = plan_new_velocities();
    planned_base_vel_ = Aplanned_base_vel_;
    planned_gripper_vel_ = Aplanned_gripper_vel_;
    planned_gripper_dq_ = Aplanned_gripper_dq_;

    // make relative to base transform
    tf::Transform base_no_trans_new = currentBaseTransform_;
    base_no_trans_new.setOrigin(tf::Vector3(0.0,0.0,0.0));
    planned_base_vel_ = base_no_trans_new.inverse() * planned_base_vel_;
    planned_base_vel_.setRotation(tf::Quaternion(0.0,0.0,0.0,1.0));
    planned_gripper_vel_rel_ = base_no_trans_new.inverse() * planned_gripper_vel_;
    planned_gripper_vel_rel_.setRotation(tf::Quaternion(0.0,0.0,0.0,1.0));
    // planned change in gripper rotation relative to base location (no change in base rotation from planner)
    tf::Quaternion planned_gripper_dq_rel = (currentGripperTransform_.getRotation() + planned_gripper_dq_) - currentBaseTransform_.getRotation();
    // std::cout << planned_gripper_dq_.x() << ", " << planned_gripper_dq_.y() << ", " << planned_gripper_dq_.z() << ", " << planned_gripper_dq_.w() << "planned_gripper_dq_" << std::endl;
    // std::cout << planned_gripper_dq_rel.x() << ", " << planned_gripper_dq_rel.y() << ", " << planned_gripper_dq_rel.z() << ", " << planned_gripper_dq_rel.w() << "planned_gripper_dq_rel" << std::endl;

    // build the observation return
    std::vector<double> obs_vector = build_obs_vector(planned_base_vel_, planned_gripper_vel_rel_, planned_gripper_dq_rel);
    // append reward to return
    obs_vector.push_back(reward);
    obs_vector.push_back(done_ret);
    obs_vector.push_back(ik_error_count_);

    return obs_vector;
}

void DynamicSystem_observeVel::visualize_robot_pose(bool vis_now, bool ik_fail, std::string logfile){

    const std::vector<std::string> joint_names = joint_model_group->getVariableNames();
    std::vector< double > joint_values;
    kinematic_state_->copyJointGroupPositions(joint_model_group, joint_values);
    moveit_msgs::RobotTrajectory fullBodyTraj_msg;
    trajectory_msgs::JointTrajectoryPoint jointPoint;
    for (int j = 0 ; j < joint_names.size(); j++)
    {
        fullBodyTraj_msg.joint_trajectory.joint_names.push_back(joint_names[j]);
        jointPoint.positions.push_back(joint_values[j]);
    }

    fullBodyTraj_msg.joint_trajectory.points.push_back(jointPoint);
    // Base Trajectory
    fullBodyTraj_msg.multi_dof_joint_trajectory.header.frame_id = "odom_combined";
    fullBodyTraj_msg.multi_dof_joint_trajectory.joint_names.push_back("world_joint");
    trajectory_msgs::MultiDOFJointTrajectoryPoint basePoint;
    geometry_msgs::Transform transform;
    tf::transformTFToMsg(currentBaseTransform_,transform);
    transform.translation.z = 0;
    basePoint.transforms.push_back(transform);
    fullBodyTraj_msg.multi_dof_joint_trajectory.points.push_back(basePoint);

    display_trajectory_.trajectory.push_back(fullBodyTraj_msg);

    if(vis_now){
        // publish trajectory
        traj_visualizer_.publish(display_trajectory_);

        // Visualize the current gripper goal
        visualization_msgs::Marker marker;
        marker.header.frame_id = "odom_combined";
        marker.header.stamp = ros::Time();
        marker.ns = "gripper_goal";
    
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = currentGripperGOAL_.getOrigin().x();
        marker.pose.position.y = currentGripperGOAL_.getOrigin().y();
        marker.pose.position.z = currentGripperGOAL_.getOrigin().z();
        marker.pose.orientation.x = currentGripperGOAL_.getRotation().x();
        marker.pose.orientation.y = currentGripperGOAL_.getRotation().y();
        marker.pose.orientation.z = currentGripperGOAL_.getRotation().z();
        marker.pose.orientation.w = currentGripperGOAL_.getRotation().w();
        marker.scale.x = 0.1;
        marker.scale.y = 0.025;
        marker.scale.z = 0.025;
        marker.color.a = 1.0; 
        if (ik_fail){
            marker.color.r = 1.0;
            marker.color.g = 0.0;    
        }   
        else{
            marker.color.r = 0.0;
            marker.color.g = 1.0;    
        }   
    
        marker.color.b = 0.0;
        
        marker_counter_++;
        if(marker_counter_ > 100)
            marker_counter_ = 0;
        marker.id = marker_counter_;
        gripper_visualizer_.publish(marker);

        // Store in rosbag
        if (logfile != ""){
            ros::Time timeStamp = ros::Time::now();
            if (timeStamp.toNSec() == 0) timeStamp = ros::TIME_MIN;

            rosbag::Bag bag;
            bag.open(logfile, rosbag::bagmode::Write);
            bag.write("traj_visualizer", timeStamp, display_trajectory_);
            bag.write("gripper_goal_visualizer", timeStamp, marker);
            bag.close();
        }
    }

}
