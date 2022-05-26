#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <modulation_rl/dynamic_system_observeVel.h>



class DynamicSystem_observeVel_Wrap:  public DynamicSystem_observeVel
{
private:
  
public:
  DynamicSystem_observeVel_Wrap(bool rnd_start_pose, uint32_t seed);

  std::vector<double> simulate_env_step_wrap(int max_allow_ik_errors, double penalty_scaling, double modulation_alpha, double modulation_lambda1, double modulation_lambda2, double modulation_lambda3);

  std::vector<double>  reset_wrap(); 

  void visualize(bool visualize_now, bool ik_fail, std::string logfile);

};