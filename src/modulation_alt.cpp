#include <dynamic_modulation/modulation.h>

using namespace cv;
using namespace std;
namespace modulation {


  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

  Modulation::Modulation(Eigen::Vector3d& curr_position, Eigen::VectorXf& curr_speed) :
  modulation_(2,2),
  modulation_gripper_(2,2),
  speed_(curr_speed),
  position_(curr_position),
  gripper_position_(7)
  {
    modulation_ << 1,0,0,1;
    gripper_position_ << 0.0,0.0,0.0,0.0,0.0,0.0,1.0;
  }

  Modulation::Modulation() :
  modulation_(2,2), 
  modulation_gripper_(2,2),
  speed_(3),
  position_(3),
  gripper_position_(7)
  {
    speed_ << 1, 1, 1;
    position_ << 1, 1, 1;
    modulation_ << 1,0,0,1;
    gripper_position_ << 0.0,0.0,0.0,0.0,0.0,0.0,1.0;


    do_ir_modulation_ = true;
    if (do_ir_modulation_)
      first_ellipse_ = 0;
    else
      first_ellipse_ = 2;

    // test to train kNN for orientation on startup
    // x/y Data
    ROS_INFO("Load Data for k-NN");
    CV_EXPORTS::CvMLData mlDataTrain;
    mlDataTrain.read_csv("/home/twelsche/Desktop/IROS_18/trainData.csv");
    const CvMat* Mat_Train_Data = mlDataTrain.get_values();
    // Data for mean angle
    CV_EXPORTS::CvMLData mlDataAngleResponse;
    mlDataAngleResponse.read_csv("/home/twelsche/Desktop/IROS_18/responseAngle0Data.csv");
    const CvMat* Mat_ResponseAngle_Data = mlDataAngleResponse.get_values();
    // Data for angle aperture
    ROS_INFO("Samples for k-NN loaded");
    CV_EXPORTS::CvMLData mlDataResponse;
    mlDataResponse.read_csv("/home/twelsche/Desktop/IROS_18/responseApertureData.csv");
    const CvMat* Mat_Response_Data = mlDataResponse.get_values();
    ROS_INFO("Aperture Data for k-NN loaded");

    const Mat& sampleIdx=Mat();
    knnAperture_.train(Mat_Train_Data, Mat_Response_Data,sampleIdx,true,32,false);
    knnAngle_.train(Mat_Train_Data, Mat_ResponseAngle_Data,sampleIdx,true,32,false);

    // Gaussian Process regression Model for IR Ellipses
    // int n=4000, m=1000;
    double tss = 0, error, f, y;
    // initialize Gaussian process for 2-D input using the squared exponential 
    // covariance function with additive white noise.

    gp_radiiX_outer.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_radiiY_outer.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_centerX_outer.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_centerY_outer.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_phi_cos_outer.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_phi_sin_outer.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_radiiX_inner.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_radiiY_inner.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_centerX_inner.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_centerY_inner.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_phi_cos_inner.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_phi_sin_inner.reset(new libgp::GaussianProcess(2, "CovSEiso"));
    gp_phi_inner.reset(new libgp::GaussianProcess(2, "CovPeriodic"));
    // initialize hyper parameter vector
    Eigen::VectorXd params(gp_radiiX_outer->covf().get_param_dim());
    Eigen::VectorXd params_per(gp_phi_inner->covf().get_param_dim());

    params << -2.28305,-0.783245;//, 0.0000000010;
    params_per << -2.28305,-0.783245, M_PI/2;
    // set parameters of covariance function
    gp_radiiX_outer->covf().set_loghyper(params);
    gp_radiiY_outer->covf().set_loghyper(params);
    gp_centerX_outer->covf().set_loghyper(params);
    gp_centerY_outer->covf().set_loghyper(params);
    gp_phi_cos_outer->covf().set_loghyper(params);
    gp_phi_sin_outer->covf().set_loghyper(params);
    gp_radiiX_inner->covf().set_loghyper(params);
    gp_radiiY_inner->covf().set_loghyper(params);
    gp_centerX_inner->covf().set_loghyper(params);
    gp_centerY_inner->covf().set_loghyper(params);
    gp_phi_cos_inner->covf().set_loghyper(params);
    gp_phi_sin_inner->covf().set_loghyper(params);
    gp_phi_inner->covf().set_loghyper(params_per);

    ROS_INFO("Params before opt: (%g,%g)", gp_radiiX_outer->covf().get_loghyper()(0), gp_radiiX_outer->covf().get_loghyper()(1));
    ROS_INFO("Params before opt: (%g,%g,%g)", gp_phi_inner->covf().get_loghyper()(0), gp_phi_inner->covf().get_loghyper()(1),gp_phi_inner->covf().get_loghyper()(2));
    // gp_phi_inner->covf().set_loghyper(params);
    // Load training data
    CV_EXPORTS::CvMLData gpDataEllipses;
    gpDataEllipses.read_csv("/home/twelsche/Desktop/IROS_18/WithACM_all_angles/ellipses_csv/EllipseDataOut.csv");
    const CvMat* Mat_gpDataEllipses = gpDataEllipses.get_values();
    for(int i = 0; i<Mat_gpDataEllipses->rows;i++)
    {
      // input variables
      double x[] = {cvGetReal2D(Mat_gpDataEllipses, i,0),cvGetReal2D(Mat_gpDataEllipses, i,1)};
      // outer ellipse
      if (cvGetReal2D(Mat_gpDataEllipses, i,4)>0.0)
      {
        double radiiX_outer = cvGetReal2D(Mat_gpDataEllipses, i,4);
        double radiiY_outer = cvGetReal2D(Mat_gpDataEllipses, i,5);
        double centerX_outer = cvGetReal2D(Mat_gpDataEllipses, i,2);
        double centerY_outer = cvGetReal2D(Mat_gpDataEllipses, i,3);
        double phi_outer = cvGetReal2D(Mat_gpDataEllipses, i,6);
        gp_radiiX_outer->add_pattern(x, radiiX_outer);
        gp_radiiY_outer->add_pattern(x, radiiY_outer);
        gp_centerX_outer->add_pattern(x, centerX_outer);
        gp_centerY_outer->add_pattern(x, centerY_outer);
        gp_phi_cos_outer->add_pattern(x, cos(phi_outer));
        gp_phi_sin_outer->add_pattern(x, sin(phi_outer));
      }
      // inner ellipse
      if (cvGetReal2D(Mat_gpDataEllipses, i,9)>0.0)
      {
        double radiiX_inner = cvGetReal2D(Mat_gpDataEllipses, i,9);
        double radiiY_inner = cvGetReal2D(Mat_gpDataEllipses, i,10);
        double centerX_inner = cvGetReal2D(Mat_gpDataEllipses, i,7);
        double centerY_inner = cvGetReal2D(Mat_gpDataEllipses, i,8);
        double phi_inner = cvGetReal2D(Mat_gpDataEllipses, i,11);
        gp_radiiX_inner->add_pattern(x, radiiX_inner);
        gp_radiiY_inner->add_pattern(x, radiiY_inner);
        gp_centerX_inner->add_pattern(x, centerX_inner);
        gp_centerY_inner->add_pattern(x, centerY_inner);
        gp_phi_cos_inner->add_pattern(x, cos(phi_inner));
        gp_phi_sin_inner->add_pattern(x, sin(phi_inner));
        gp_phi_inner->add_pattern(x, phi_inner);
      }

      // ROS_INFO("Outer Ellipse %d: (%g, %g, %g, %g, %g)",i,radiiX_outer,radiiY_outer,centerX_outer,centerY_outer,phi_outer);
      // ROS_INFO("Inner Ellipse %d: (%g, %g, %g, %g, %g)",i,radiiX_inner,radiiY_inner,centerX_inner,centerY_inner,phi_inner);
      
      
      // gp_phi_inner->add_pattern(x, phi_inner);
    } 
    // Hyper-parameter optimization
    libgp::RProp rprop;
    rprop.init();
    rprop.maximize(&(*gp_radiiX_outer), 500, 0);
    rprop.maximize(&(*gp_radiiY_outer), 500, 0);
    rprop.maximize(&(*gp_centerX_outer), 500, 0);
    rprop.maximize(&(*gp_centerY_outer), 500, 0);
    rprop.maximize(&(*gp_phi_cos_outer), 500, 0);
    rprop.maximize(&(*gp_phi_sin_outer), 500, 0);
    rprop.maximize(&(*gp_radiiX_inner), 500, 0);
    rprop.maximize(&(*gp_radiiY_inner), 500, 0);
    rprop.maximize(&(*gp_centerX_inner), 500, 0);
    rprop.maximize(&(*gp_centerY_inner), 500, 0);
    rprop.maximize(&(*gp_phi_cos_inner), 500, 0); 
    rprop.maximize(&(*gp_phi_sin_inner), 500, 0);
    rprop.maximize(&(*gp_phi_inner), 500, 0);
    ROS_INFO("Params after opt: (%g,%g)", gp_radiiX_outer->covf().get_loghyper()(0), gp_radiiX_outer->covf().get_loghyper()(1));
    ROS_INFO("ParamsPer after opt: (%g,%g,%g)", gp_phi_inner->covf().get_loghyper()(0), gp_phi_inner->covf().get_loghyper()(1),gp_phi_inner->covf().get_loghyper()(2));

    // create test-set and evaluate GP Regression

    bool do_GP_evaluation = false;
    if (do_GP_evaluation)
    {
      int test_z_nr = 300;
      int test_pitch_nr = 300;
      double min_z = 0.3;
      double max_z = 1.3;
      double min_pitch = -1.6;
      double max_pitch = 1.5;
      std::ofstream CSVFile;
      CSVFile.open("/home/twelsche/Desktop/IROS_18/WithACM_all_angles/ellipses_csv/GP_Regression_ellipses.csv", std::ios::ate | std::ios::app);
      // CSVFile.open("IRM_test.csv", std::ios::ate | std::ios::app);
      //insert title row 
      CSVFile << "z_gripper" << "," << "pitch_gripper" << "," << "gp_radiiX_outer" << "," << "gp_radiiY_outer" << "," << "gp_centerX_outer" << "," << "gp_centerY_outer" << "," << "gp_phi_outer" 
                                                       << "," << "gp_radiiX_inner" << "," << "gp_radiiY_inner" << "," << "gp_centerX_inner" << "," << "gp_centerY_inner" << "," << "gp_phi_inner"<<"\n";
      for(int i = 0; i < test_z_nr; i++)
      {
        for(int j = 0; j < test_pitch_nr; j++)
        {
          double x_test[] = {min_z+ ((double)i/(double)test_z_nr)*(max_z-min_z),min_pitch+ ((double)j/(double)test_pitch_nr)*(max_pitch-min_pitch)};
          double cos_alpha_outer = gp_phi_cos_outer->f(x_test);
          double sin_alpha_outer = gp_phi_sin_outer->f(x_test);
          double alpha_outer = atan2(sin_alpha_outer,cos_alpha_outer);        
            
          double cos_alpha_inner = gp_phi_cos_inner->f(x_test);
          double sin_alpha_inner = gp_phi_sin_inner->f(x_test);
          double alpha_inner = atan2(sin_alpha_inner,cos_alpha_inner);//gp_phi_inner->f(x_test);//
          
          CSVFile << x_test[0] << "," << x_test[1] 
                  << "," << gp_radiiX_outer->f(x_test) << "," << gp_radiiY_outer->f(x_test) << "," << gp_centerX_outer->f(x_test) << "," << gp_centerY_outer->f(x_test) << "," << alpha_outer 
                  << "," << gp_radiiX_inner->f(x_test) << "," << gp_radiiY_inner->f(x_test) << "," << gp_centerX_inner->f(x_test) << "," << gp_centerY_inner->f(x_test) << "," << alpha_inner <<"\n";
          // f = gp_radiiX_outer.f(x_test);
        }
      }
      ROS_INFO("GP Regression test created");
    }




  }

  Modulation::~Modulation()
  {

  }

  std::vector<ellipse_extraction::Ellipse>& Modulation::getEllipses() {
    return ellipses_;
  }

 std::vector<ellipse_extraction::Ellipse>& Modulation::getEllipses(Eigen::Vector3d& curr_pose, Eigen::VectorXf& curr_speed,Eigen::VectorXd& curr_gripper_pose,double dt) {
    // updateSpeedAndPosition(curr_pose,curr_speed,curr_gripper_pose,dt);
    return ellipses_;
  }

   void Modulation::setEllipses() {
    std::vector<ellipse_extraction::Ellipse> ellipses;

    ellipses.push_back(ellipse_extraction::Ellipse(gripper_position_[0],gripper_position_[1],"inner"));
    ellipses.push_back(ellipse_extraction::Ellipse(gripper_position_[0],gripper_position_[1],"outter"));
    // ellipses.push_back(ellipse_extraction::Ellipse(1.565,-1.3,M_PI/2,1.18));// (1.515,-1.3,M_PI/2,1.18));
    // ellipses.push_back(ellipse_extraction::Ellipse(1.54,1.237,M_PI/2,1.20814));//(1.49,1.187,M_PI/2,1.20814));
    ellipses.push_back(ellipse_extraction::Ellipse(4.69,0.67,-0.7,0.7)); // Kallax
    // ellipses.push_back(ellipse_extraction::Ellipse(4.1,0.8,-0.7,0.7)); // pose1
    ellipses.push_back(ellipse_extraction::Ellipse(4.3,-0.6,-0.7,0.7)); //pose2
    ellipses.push_back(ellipse_extraction::Ellipse(4.1,-0.5,-0.7,0.7)); //pose2
    ellipses_ = ellipses;
    ROS_INFO("ellipses set");
    
    
  }


  void Modulation::updateSpeedAndPosition(Eigen::Vector3d& curr_pose, Eigen::VectorXf& curr_speed,Eigen::VectorXd& curr_gripper_pose,double dt) {
    position_ = curr_pose;
    speed_ = curr_speed;
    gripper_position_ = curr_gripper_pose;
    Eigen::Matrix2f R(2,2);

    Eigen::Isometry3d gripperPose;
    gripperPose.setIdentity();
    Eigen::Quaterniond Q = Eigen::Quaterniond(curr_gripper_pose(6),curr_gripper_pose(3),curr_gripper_pose(4),curr_gripper_pose(5));
    gripperPose.linear() = Q.matrix();
    double gripper_pitch;
    Eigen::Vector3d euler = Q.toRotationMatrix().eulerAngles(2, 1, 0);
    gripper_pitch = euler[1]; 
    if (gripper_pitch > M_PI/2)
      gripper_pitch = M_PI -gripper_pitch;
    else if (gripper_pitch < -M_PI/2)
      gripper_pitch = -M_PI -gripper_pitch;

    // if (gripper_pitch > M_PI/3)
    //   gripper_pitch = M_PI/3;
    // if (gripper_pitch < -M_PI/3)
    //   gripper_pitch = -M_PI/3;
    gripper_yaw_speed_ = speed_(6);
    // gripper_pitch =0.0;

    
    
    
    for (int k = 0 ; k < ellipses_.size(); k++) {
      if(ellipses_[k].getType() == "outter" || ellipses_[k].getType() == "inner")
      {
        // retrieve parameters for ir ellipses from gripper (x, pitch)
        
        Eigen::Vector3d x_Offset_gripper;
        x_Offset_gripper << -0.18, 0.0, 0.0;
        x_Offset_gripper = gripperPose.linear() * x_Offset_gripper;
        Eigen::Vector3d wrist_pose;
        wrist_pose << gripper_position_[0] + x_Offset_gripper[0], gripper_position_[1] + x_Offset_gripper[1], gripper_position_[2] + x_Offset_gripper[2]-0.1;
        // double relative_Z_gripper_base = wrist_pose(2) +(0.7467-current_base_pose[2]);
        double x_test[] = {wrist_pose(2),gripper_pitch};
        // double x_test[] = {0.8,0.0};
        // update speed and position of irm ellipses
        
        Eigen::Vector3d radial_velocity;
        Eigen::Vector3d angle_velocity;
        angle_velocity << curr_speed[3],curr_speed[4],curr_speed[5];        
        std::vector<double> ellipse_speed; 
        if (ellipses_[k].getType() == "inner")
        {
          ellipses_[k].setHeight(gp_radiiY_inner->f(x_test)+0.05);
          ellipses_[k].setWidth(gp_radiiX_inner->f(x_test)+0.05);
          Eigen::Vector3d xOffset_inner;
          xOffset_inner << gp_centerX_inner->f(x_test) , gp_centerY_inner->f(x_test), 0.0;
          xOffset_inner = gripperPose.linear() * xOffset_inner;
          double alpha = atan2(xOffset_inner[1], xOffset_inner[0]);
          double cos_alpha_inner = gp_phi_cos_inner->f(x_test);
          double sin_alpha_inner = gp_phi_sin_inner->f(x_test);
          double alpha_inner = atan2(sin_alpha_inner,cos_alpha_inner);//0.0;//
          double cosangle = cos(alpha-alpha_inner);
          double sinangle = sin(alpha-alpha_inner);
          R << cosangle ,-sinangle , sinangle,cosangle;
          ellipses_[k].setR(R); 
          ellipses_[k].setPPoint(wrist_pose[0]+xOffset_inner[0], wrist_pose[1]+xOffset_inner[1]);
          radial_velocity = angle_velocity.cross(x_Offset_gripper);
          ellipse_speed.push_back(curr_speed[0] + radial_velocity[0]);
          ellipse_speed.push_back(curr_speed[1] + radial_velocity[1]);
        }
        else
        {
          ellipses_[k].setHeight(gp_radiiX_outer->f(x_test)+0.0);
          ellipses_[k].setWidth(gp_radiiY_outer->f(x_test)+0.0);
          Eigen::Vector3d xOffset_outer;
          xOffset_outer << gp_centerX_outer->f(x_test), gp_centerY_outer->f(x_test), 0.0;
          xOffset_outer = gripperPose.linear() * xOffset_outer;
          double alpha = atan2(xOffset_outer[1], xOffset_outer[0]);
          double cos_alpha_outer = gp_phi_cos_outer->f(x_test);
          double sin_alpha_outer = gp_phi_sin_outer->f(x_test);
          double alpha_outer = atan2(sin_alpha_outer,cos_alpha_outer);
          double cosangle = cos(alpha-alpha_outer);
          double sinangle = sin(alpha-alpha_outer);
          R << cosangle ,-sinangle , sinangle,cosangle;
          ellipses_[k].setR(R); 
          ellipses_[k].setPPoint(wrist_pose[0]+xOffset_outer[0], wrist_pose[1]+xOffset_outer[1]);
          radial_velocity =  angle_velocity.cross(x_Offset_gripper);
          ellipse_speed.push_back(curr_speed[0] + radial_velocity[0]);
          ellipse_speed.push_back(curr_speed[1] + radial_velocity[1]);
          // ROS_INFO("X-Test for GP Regression: (%g, %g): outer (%g,%g,%g,%g)",x_test[0],x_test[1],gp_radiiX_outer->f(x_test),
          //          gp_radiiY_outer->f(x_test),gp_centerX_outer->f(x_test),gp_centerY_outer->f(x_test));

          // update orientation part of positioning for irm ellipses
          int nr_neighbors = 19;
          Eigen::Vector2f pos_ell_frame;
          pos_ell_frame << position_[0] -ellipses_[k].getPPoint()[0] , position_[1] -ellipses_[k].getPPoint()[1];
          pos_ell_frame = ellipses_[k].getR().transpose()*pos_ell_frame;
          float _sample[4];
          CvMat sample_beta0 = cvMat( 1, 4, CV_32FC1, _sample );
          sample_beta0.data.fl[0] = (float)curr_gripper_pose(2);
          sample_beta0.data.fl[1] = (float)Q.toRotationMatrix().eulerAngles(2, 1, 0)[1];//0.0;
          if(sample_beta0.data.fl[1] >M_PI/2)
            sample_beta0.data.fl[1] = M_PI - sample_beta0.data.fl[1];
          else if(sample_beta0.data.fl[1] < - M_PI/2)
            sample_beta0.data.fl[1] = -M_PI - sample_beta0.data.fl[1];
          sample_beta0.data.fl[2] = (float)pos_ell_frame[0];
          sample_beta0.data.fl[3] = (float)pos_ell_frame[1];
          float _response[nr_neighbors];
          float _neighbors[1];
          CvMat resultMat = cvMat(1,1,CV_32FC1,_neighbors);
          CvMat neighborResponses = cvMat(1,nr_neighbors,CV_32F,_response);
          const float **neighbors=0;
          float result_beta0 = knnAngle_.find_nearest(&sample_beta0, nr_neighbors,&resultMat,neighbors,&neighborResponses);
          double sum_sin = 0.0;
          double sum_cos = 0.0;
          for (int s = 0; s < nr_neighbors; s++)
          {
            double neighbor_i = (double) neighborResponses.data.fl[s];
            sum_sin += sin(neighbor_i);
            sum_cos += cos(neighbor_i);
          }
          result_beta0 = atan2(sum_sin,sum_cos);

          // find aperture for leagal orientation with knn regression
          float result_beta_ap = knnAperture_.find_nearest(&sample_beta0, nr_neighbors); 
          
          ellipses_[k].setPPointAlpha(result_beta0);
          ellipses_[k].setAlphaAp(result_beta_ap);
        }  
        ellipses_[k].setSpeed(ellipse_speed);        
      }
    }
  }

  void Modulation::computeXiWave() {
    xi_wave_.clear();
    for (int i = 0; i < ellipses_.size(); i++) {
      Eigen::Vector2f pos_ell_frame;
      pos_ell_frame << position_[0] -ellipses_[i].getPPoint()[0] , position_[1] -ellipses_[i].getPPoint()[1];
      pos_ell_frame = ellipses_[i].getR().transpose()*pos_ell_frame;
      std::vector<double> xi_wave_i = {pos_ell_frame[0] , pos_ell_frame[1]};
      xi_wave_.push_back(xi_wave_i);
    }
  }

  void Modulation::computeGammaAlpha(int ellipseNr){
    //robot base orientation
    Eigen::Matrix2f R;
    R << std::cos(position_[2]),-std::sin(position_[2]),std::sin(position_[2]),std::cos(position_[2]);
    Eigen::Vector2f or_x;
    or_x << 1.0,0.0;
    or_x = ellipses_[ellipseNr].getR().transpose()*R*or_x; // 
    // angle between robot base orientation and connection to gripper
    double dot = -xi_wave_[ellipseNr][0]*or_x[0] - xi_wave_[ellipseNr][1]*or_x[1];     // dot product between [x1, y1] and [x2, y2]
    double det = -xi_wave_[ellipseNr][1]*or_x[0] + xi_wave_[ellipseNr][0]*or_x[1];     //x1*y2 - y1*x2      # determinant, angle = atan2(det, dot)
    double current_alpha = atan2(det, dot);

    //base orientation in ellipseframe
    double base_angle = position_[2]+ellipses_[ellipseNr].getAlpha();// atan2(or_x[1], or_x[0]);
    double alpha_dist = ellipses_[ellipseNr].getPPointAlpha() - base_angle;

    if (alpha_dist < -M_PI)
    {
      alpha_dist += 2.0*M_PI;
    }
    else if (alpha_dist > M_PI)
      alpha_dist -= 2.0*M_PI;
    
    double powerF = 2.0;
    gamma_alpha_ = pow((alpha_dist/ellipses_[ellipseNr].getAlphaAp()/2.0), powerF);
    double speed_before = speed_(12);//-gripper_yaw_speed_;
    
    if (gamma_alpha_ >= 0.1) 
    {
      // ROS_INFO("Outside Angle Bound. Angledist to opt: %g, measured angle: %g and speed: %g",alpha_dist*180.0/M_PI,position_[2]*180.0/M_PI,speed_(12));
      if(alpha_dist*speed_before>=0.0)
        speed_before = speed_before *gamma_alpha_*10;
      else
        speed_before = -speed_before *gamma_alpha_*10;
    }
    else
    {
      // ROS_INFO("Inside  Angle Bound. Angledist to opt: %g, measured angle: %g and speed: %g",alpha_dist*180.0/M_PI,position_[2]*180.0/M_PI,speed_(12));
      if(alpha_dist*speed_before<0.0)
        speed_before = speed_before *(1.0-gamma_alpha_);
    }
    if(do_ir_modulation_)
      speed_(12) = speed_before;// + gripper_yaw_speed_;
  }


  void Modulation::computeGamma() {
    gamma_.clear();
    computeXiWave();
    int i = 0;
    collision_ = false;
    for (ellipse_extraction::Ellipse ellipse : ellipses_) {
      double gamma_i = pow(pow((xi_wave_[i][0]/ellipse.getHeight()), 2*ellipse.getP1()) + pow((xi_wave_[i][1]/ellipse.getWidth()), 2*ellipse.getP2()),1.0/ellipse.getP2());
      ellipses_[i].setInCollision(false);
      if(ellipse.getType() == "outter")
      {
        gamma_i = 1.0/gamma_i;
        computeGammaAlpha(i);
      }  
      // ROS_INFO("Ellipse %d of %d (%s), with gamma_i: (%lf)", i + 1, int(ellipses_.size()),ellipse.getType().c_str(), gamma_i);
      real_gamma_.push_back(gamma_i);      
      if(gamma_i < 1.0) {//1.00015) {//
        ellipses_[i].setInCollision(true);        
        // ROS_INFO("Something wrong. Inside obstacle: %d of %d (%s), gamma_i: (%lf)", i + 1, int(ellipses_.size()),ellipse.getType().c_str(), gamma_i);
        if(ellipse.getType() == "outter")
          gamma_i = 1.0;//1.00015;//  
        else if (ellipse.getType() == "inner")
        {
          gamma_i = 1.0;//1.00015;//
          collision_ = true;
        }
        else
          gamma_i = 1.0;
      }
      gamma_.push_back(gamma_i);

      ellipse.setGamma(gamma_i);
      i++;
    }
  }

  double Modulation::computeWeight(int k)
  {
    double w = 1;
    for (int i = first_ellipse_; i < ellipses_.size(); i++) {
      if (i != k) {
        w = w * ((gamma_[i] - 1)/((gamma_[k] - 1) + (gamma_[i] - 1)));
      }
    }  
    if(w != w)
    {
      w = 1.0;
      for (int i = first_ellipse_; i < ellipses_.size(); i++) {
        if (i != k) {
          w = w * ((real_gamma_[i] - 1)/((real_gamma_[k] - 1) + (real_gamma_[i] - 1)));
        }
      } 
    }
    if(!do_ir_modulation_ & first_ellipse_>k)
      w = 0;
    return w;
  }

  std::vector<double> Modulation::computeEigenvalue(int k) {
    std::vector<double> lambda;
    double w = computeWeight(k);
    double collision_repulsion = -50.0;

    Eigen::Vector2f speed;
    speed << speed_(7)-ellipses_[k].getSpeed()[0],speed_(8)-ellipses_[k].getSpeed()[1];
    Eigen::Vector2f e_k1;
    e_k1 << assembleE_k(k)(0,0),assembleE_k(k)(0,1);
    bool passed_object = false;
    if(speed.transpose().dot(ellipses_[k].getR()*e_k1) > 0.0)
      passed_object = true; // use this to conmtroll tail effekt, stop modulation if object already passed

    // OUTER IRM Bound
    if(ellipses_[k].getType() == "outter")
    {
      if(passed_object && !ellipses_[k].getInCollision())
        lambda = {1.0 - (w / pow(gamma_[k],1.0/ellipses_[k].getRho())), 1.0 };   
      else if (passed_object && ellipses_[k].getInCollision())
        lambda = {collision_repulsion, 1.0 };
      else
        lambda = {1.0,1.0};
    }
    // INNER IRM Bound
    else if (ellipses_[k].getType() == "inner")
    {
      lambda = {1.0 - (w / pow(gamma_[k],1.0/ellipses_[k].getRho())), 1.0 + 1.0/500000.0*(w / pow(gamma_[k],1.0/ellipses_[k].getRho()))};
      if(passed_object && !ellipses_[k].getInCollision())
      {
        lambda[0] = 1.0;
        lambda[1] = 1.0;
      }
      // case of colliding with bound
      else if (ellipses_[k].getInCollision())
      {
        if (passed_object)
          lambda[0] = 2 - lambda[0]; // inside bound but moving out -> accelerate
        else 
          lambda[0] = collision_repulsion; // inside bound and moving further in -> mirror and increase velocity
        lambda[1] = 1.0 + (w / pow(gamma_[k],1.0/ellipses_[k].getRho()));
      }
    }
    // Other Obstacles
    else
    {
      lambda = {1.0 - (w / pow(gamma_[k],1.0/ellipses_[k].getRho())), 1.0 + (w / pow(gamma_[k],1.0/ellipses_[k].getRho()))};
      if(passed_object && !ellipses_[k].getInCollision())
      {
        lambda[0] = 1.0;
        lambda[1] = 1.0;
      }
      else if (ellipses_[k].getInCollision())
      {
        if (passed_object)
          lambda[0] = 2 - lambda[0];
        else 
          lambda[0] = collision_repulsion;
      } 
      else if(gamma_[k]<1.01 && !ellipses_[k].getInCollision())
      {
        // speed_(0) = speed_(0)*(gamma_[k]-1.0);
        // speed_(1) = speed_(1)*(gamma_[k]-1.0);
        // speed_(2) = speed_(2)*(gamma_[k]-1.0);
      }     
    }

    // if(gamma_[k]<1.01 && !ellipses_[k].getInCollision())
    //   {
    //     speed_(0) = speed_(0)*(gamma_[k]-1.0);
    //     speed_(1) = speed_(1)*(gamma_[k]-1.0);
    //     speed_(2) = speed_(2)*(gamma_[k]-1.0);
    //   } 
    if (ellipses_[k].getInCollision() && do_ir_modulation_)
    {
        speed_(0) = 0.0;
        speed_(1) = 0.0;
        speed_(2) = 0.0;
    }
    if(!do_ir_modulation_ & first_ellipse_>k)
      lambda = {1.0,1.0};
    return lambda;
  }

  Eigen::MatrixXf Modulation::assembleD_k(int k) {
    Eigen::MatrixXf d_k(2,2);
    d_k.setIdentity();
    std::vector<double> lambda = computeEigenvalue(k);

    for(int i = 0; i < 2; i++) {
      d_k(i, i) = lambda[i];
    }
    return d_k;
  }

  std::vector<double>  Modulation::computeHyperplane(int k) {
    //Derivation of Gamma in ~Xi_i direction
    std::vector<double> n = {(pow(xi_wave_[k][0] / ellipses_[k].getHeight(), 2.0*ellipses_[k].getP1() -1))*2*ellipses_[k].getP1() /ellipses_[k].getHeight(),
      (pow(xi_wave_[k][1] / ellipses_[k].getWidth(), 2.0*ellipses_[k].getP2() -1)) *2*ellipses_[k].getP2() /ellipses_[k].getWidth()};

    Eigen::Vector2f n_rot;
    n_rot << n[0],n[1];
    n_rot = ellipses_[k].getR()*n_rot;
    ellipses_[k].setHyperNormal(std::vector<double> {n_rot(0),n_rot(1)} );
    return n;
  };


  std::vector<std::vector<double> > Modulation::computeEBase(int k, std::vector<double> normal) {
    int d = 2;
    std::vector<std::vector<double> > base = {{0, 0}};
    for (int i = 1; i <= d - 1; i++) {
      for (int j = 1; j <= d; j++) {
        if (j == 1) {
          base[i-1][j-1] = -normal[i - 1];
        } else if (j == i && i != 1) {
          base[i-1][j-1] = normal[0];
        } else {
          base[i-1][j-1] = 0;
        }
      }
    }
    return base;
  };



  Eigen::MatrixXf Modulation::assembleE_k(int k) {
    Eigen::MatrixXf e_k(2,2);
    std::vector<double> norm = computeHyperplane(k);
    std::vector<std::vector<double> > base = computeEBase(k, norm);
    for (int i = 0; i < 2; i++) {
      e_k(i, 0) = norm[i];
      if(i==0)
        e_k(i, 1) = norm[1];
      else
        e_k(i, 1) = -norm[0];
    }
    return e_k;
  }

  void Modulation::computeModulationMatrix() {
    modulation_ << 1, 0,
                   0, 1;
    std::stringstream mes;
    computeGamma();
    bool out_mod = false;
    for (int k = 0; k < ellipses_.size(); k++) {
      Eigen::MatrixXf d_k = assembleD_k(k);
      Eigen::MatrixXf e_k = assembleE_k(k);
      Eigen::MatrixXf res = (ellipses_[k].getR()*e_k * d_k * e_k.inverse()*ellipses_[k].getR().transpose());
      modulation_ = (res * modulation_);
    }

  }

  Eigen::VectorXf Modulation::compModulation() {
    if (ellipses_.size() == 0)
      return speed_;
    computeModulationMatrix();
    Eigen::VectorXf d2(2);
    // find weighted relative speed with respect to obstacles
    double meanVelX = 0.0;
    double meanVelY = 0.0;
    double weightSum = 0.0;
    for (int k = 0; k <ellipses_.size(); k++) {
      double weight_k = computeWeight(k);
      // if (weight_k < 0.1)
      //   continue;
      weightSum += weight_k;
      meanVelX += weight_k*ellipses_[k].getSpeed()[0];
      meanVelY += weight_k*ellipses_[k].getSpeed()[1];
      // ROS_INFO("Type %s, weight: %g, Speed(%g, %g)",ellipses_[k].getType().c_str(),computeWeight(k),ellipses_[k].getSpeed()[0],ellipses_[k].getSpeed()[1]);
    }
    d2 << speed_[7]-meanVelX/weightSum, speed_[8]-meanVelY/weightSum;
    d2 = modulation_ * d2;
    speed_(7) = d2[0]+meanVelX/weightSum;
    speed_(8) = d2[1]+meanVelY/weightSum;

    return speed_;
  }

  Eigen::VectorXf Modulation::run(Eigen::VectorXf& curr_pose, Eigen::VectorXf& curr_speed,double dt) {

    Eigen::Vector3d trans;
    trans[0] = curr_pose(7);
    trans[1] = curr_pose(8);
    // tf::Quaternion q(curr_pose(10),curr_pose(11),curr_pose(12),curr_pose(13));
    // trans[2] = tf::getYaw(q);
    Eigen::Quaterniond Q2 = Eigen::Quaterniond(curr_pose(13),curr_pose(10),curr_pose(11),curr_pose(12));
    auto euler = Q2.toRotationMatrix().eulerAngles(0, 1, 2);
    trans[2] = euler[2];
    Eigen::VectorXd curr_gripper_pose(7);
    curr_gripper_pose[0] = curr_pose(0);
    curr_gripper_pose[1] = curr_pose(1);
    curr_gripper_pose[2] = curr_pose(2);
    curr_gripper_pose[3] = curr_pose(3);
    curr_gripper_pose[4] = curr_pose(4);
    curr_gripper_pose[5] = curr_pose(5);
    curr_gripper_pose[6] = curr_pose(6); 
    // Update speed and position for the irm objects
    updateSpeedAndPosition(trans, curr_speed,curr_gripper_pose, dt);
    // compute and return modulated velocity
    return compModulation();;
  }


  double computeL2Norm(std::vector<double> v) {
    double res = 0;
    for(double entry : v) {
      res += entry * entry;
    }
    return sqrt(res);
  }



}
