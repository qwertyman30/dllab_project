#include <modulation_rl/dynamic_system_observeVel_py_wrap.h>

//int add(int i, int j) {
//    return i + j;
//}

DynamicSystem_observeVel_Wrap::DynamicSystem_observeVel_Wrap(bool rnd_start_pose, uint32_t seed):
    DynamicSystem_observeVel(rnd_start_pose, seed)
    {

    };
    

std::vector<double> DynamicSystem_observeVel_Wrap::simulate_env_step_wrap(int max_allow_ik_errors, double penalty_scaling, double modulation_alpha, double modulation_lambda1, double modulation_lambda2, double modulation_lambda3){
    return simulate_env_step(max_allow_ik_errors, penalty_scaling, modulation_alpha, modulation_lambda1, modulation_lambda2, modulation_lambda3);
  };

std::vector<double>  DynamicSystem_observeVel_Wrap::reset_wrap(){return reset();};

void DynamicSystem_observeVel_Wrap::visualize(bool visualize_now, bool ik_fail, std::string logfile){
      visualize_robot_pose(visualize_now, ik_fail, logfile);
};


namespace py = pybind11;

PYBIND11_MODULE(dynamic_system_observeVel_py, m) {
    
    py::class_<DynamicSystem_observeVel_Wrap>(m, "DynamicSystem_observeVel_Wrap")
        .def(py::init<bool, uint32_t>())
        .def("simulate_env_step_wrap", &DynamicSystem_observeVel_Wrap::simulate_env_step_wrap, "Execute the next time step in environment.")
        .def("reset_wrap", &DynamicSystem_observeVel_Wrap::reset_wrap, "Reset environment.")
        .def("visualize", &DynamicSystem_observeVel_Wrap::visualize, "Visualize trajectory.");
        

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
