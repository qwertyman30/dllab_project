import numpy as np
import torch

from modulation_rl.ModulationEnv import ModulationEnv


class Config:
    ik_fail_thresh = 100
    ik_fail_thresh_eval = 100
    # feel free to also directly adjust the form of the reward function in the c++ code
    # only suggestion is to keep it simple. Too much reward engineering does not generalise well
    penalty_scaling = 0.1
    # use random start pose to increase diffulty (but also ability to generalise)
    rnd_start_pose = False
    # suggested to keep the following values at default, but feel free to play around with
    arctan2_alpha = True
    max_action = [1.,  # alpha_x
                  1.,  # alpha_y
                  2.,  # lambda1
                  2.,  # lambda2
                  0.2,  # lambda3
                  ]
    min_action = [-1, -1, -2, -2, -0.2]
    seed = 42


def uniform_action_selection(action_dim):
    return np.random.uniform(-1, 1, action_dim)


def rnd_rollout(nsteps=25000):
    """Take random actinos through the environment"""
    # if you have a gpu, check that recognised and all requirements (cuda, cudnn) are in place
    print("Cuda available? {}".format(torch.cuda.is_available()))

    env = ModulationEnv(Config.rnd_start_pose,
                        Config.ik_fail_thresh,
                        Config.ik_fail_thresh_eval,
                        Config.penalty_scaling,
                        Config.arctan2_alpha,
                        Config.min_action,
                        Config.max_action,
                        Config.seed)

    obs = env.reset()
    for i in range(nsteps):
        action = uniform_action_selection(env.action_dim)
        obs, reward, done_return, nr_kin_failures = env.step(action)

        if done_return > 0:
            obs = env.reset()
            env.visualize(done_return > 0, 'demo_logs', 'iter{}.bag'.format(i))
            print("Episode terminated with {} kinetic failures. {}.".format(nr_kin_failures,
                                                                            env.parse_done_return(done_return)))


if __name__ == '__main__':
    rnd_rollout(nsteps=int(100000))
