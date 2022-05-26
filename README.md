# Modulation RL

## Install
The RL environment is written in C++ and connected to python through bindings.
The interplay with ROS requires python2.7.

If possible avoid Windows platforms, especially visualisations might not be possible.
Though with WSL2 (ensure you have the Windows May update installed) it might be possible now.

### ROS
Install the appropriate version for your system (full install recommended): http://wiki.ros.org/ROS/Installation

Install the corresponding catkin package for python bindings
        
    sudo apt install ros-[version]-pybind11-catkin
        
Install moveit and the pr2
    
    sudo apt-get install ros-[version]-moveit
    sudo apt-get install ros-[version]-pr2-simulator
    sudo apt-get install ros-[version]-moveit-pr2 
    
Create a catkin workspace

    mkdir ~/catkin_ws
    cd catkin_ws
    catkin init

Fork the repo and clone into `./src`
    
    cd src
    git clone [url] src/dllab_modulation_rl
    
Create the enviroment. We recommend using conda, which requires to first install Anaconda or Miniconda. Then  
    
    conda env create -f src/dllab_modulation_rl/environment.yml
    conda activate dllab_modulation_rl
    
Build the ROS package (alternative: `catkin_make`, but don't mix the two)
    
    catkin build
    
Each new build of the ROS / C++package requires a
    
    source devel/setup.bash
    
To be able to visualise install rviz

    http://wiki.ros.org/rviz/UserGuide
    

## Run
_Outside_ of the conda environment do:
1. start a roscore (launches one automatically as well but stopping the rosnode would then close any other rosnodes in the background as well)
        
        roscore
        
2. start gazebo with the pr2 robot
        
        roslaunch pr2_gazebo pr2_empty_world.launch
        
3. start moveit

        roslaunch pr2_moveit_config move_group.launch

4. Run the demo script inside the conda env:

        conda activate dllab_modulation_rl
        python src/dllab_modulation_rl/python/demo.py

5. [Only to visualise] start rviz while the environment is running:

        rviz
        
   - Select
   
            Global Options -> Fixed Frame -> odom_combined
            
   - Show trajectories (continually playing last one visualised): 
            
            Add -> Trajectory
            Trajectory -> Trajectory Topic '/TD3_PR2_ik/...'
            
   - Show last 100 target end-effector poses, colored by success:
            
            Add -> By topic -> /TD3_PR2_ik/ -> Marker


## Update 22/06/2020
- environment constructor now takes a seed (int) as input, make environment runs (almost) fully reproducible
- the observation state now includes the current joint position of the gripper arm
- ability to store recorded trajectories: pass a logdir and logfile to env.visualise to store trajectories.
These can then be replayed later on. Note that this might not be able to display the end-effector goal in its current form.
Do points 1-3 from the "Run" instructions, but without running any python script. Instead start rviz and run
        
        rosbag play [--loop] logdir/*.bag

To update the project simply 
1. pull the latest commit from this repo
2. Build the ROS package again with `catkin build`
3. Source the devel file again `source devel/setup.bash`

## Update 07/07/2020
- added a dockerfile which may or may not be useful for your setup
    - to build the image:  `docker build . -t dllab`
    - install the drivers to use gpus: https://github.com/NVIDIA/nvidia-docker
    - to run the image: `docker run --gpus all dllab` or interactive `docker run -it --gpus all dllab bash`


## Further resources
- Original approach, problem statement and details to the modulation:  
    [Coupling  Mobile  Base  and  End-Effector  Motion  in  Task  Space](http://ais.informatik.uni-freiburg.de/publications/papers/welschehold18iros.pdf)
- [Openai intro](https://spinningup.openai.com/en/latest/user/introduction.html) to the type of algorithms


## Your part
Implement an RL agent that minimises the number of kinetic failures.

