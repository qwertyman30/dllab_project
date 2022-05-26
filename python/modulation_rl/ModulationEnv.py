import os

import numpy as np
from dynamic_system_observeVel_py import DynamicSystem_observeVel_Wrap


class ModulationEnv:
    """
    Environment for modulating the base of a mobile manipulator robot.

    Action space:
    - alpha_x    Direction of the norm vector in which to apply the velocity. Direction is set as arctan2(alpha_x, alpha_y)
    - alpha_y
    - lambda1    Velocity in direction of the norm vector. E.g. 1: no modulation, 2: double the velocity, -1: same speed in reverse direction
    - lambda2    Velocity orthogonal to norm vector. Not required if direction of the vector can be set. Input ignored and always set to 1 within the c++ code
    - lambda3    Angular velocity for the base
    """
    DoneReturnCode = {0: 'Ongoing',
                      1: 'Success: goal reached',
                      2: 'Failure: too many ik_fail or ik_fail at goal'}

    def __init__(self,
                 rnd_start_pose,
                 ik_fail_thresh,
                 ik_fail_thresh_eval,
                 penalty_scaling,
                 arctan2_alpha,
                 min_actions,
                 max_actions,
                 seed):
        """
        Args:
            rnd_start_pose: whether to start each episode from a random gripper pose
            ik_fail_thresh: number of kinetic failures after which to abort the episode
            ik_fail_thresh_eval: number of kinetic failures after which to abort the episode during evaluation
                (allows to compare across different ik_fail_thresh)
            penalty_scaling: how much to weight the penalty for large action modulations in the reward
            arctan2_alpha: whether to construct modulation_aplha as actan2 or directly learn it
                (in which case actions[1] will be ignored).
                Setting to false requires to also change the bounds for this action
            min_actions: lower bound constraints for actions
            max_actions: upper bound constraints for actions
        """
        self._env = DynamicSystem_observeVel_Wrap(rnd_start_pose, seed)
        self.state_dim = len(self._env.reset_wrap())
        self.action_dim = 5

        self._min_actions = np.array(min_actions)
        self._max_actions = np.array(max_actions)

        self._ik_fail_thresh = ik_fail_thresh
        self._ik_fail_thresh_eval = ik_fail_thresh_eval
        self._penalty_scaling = penalty_scaling
        self._arctan2_alpha = arctan2_alpha

    def _parse_env_output(self, retval):
        obs = retval[:self.state_dim]
        reward = retval[self.state_dim]
        done_return = retval[self.state_dim + 1]
        nr_kin_failures = retval[self.state_dim + 2]
        return obs, reward, done_return, nr_kin_failures

    def scale_action(self, action):
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)
        :param action: (np.ndarray) Action to scale
        :return: (np.ndarray) Scaled action
        """
        low, high = self._min_actions, self._max_actions
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: Action to un-scale
        """
        low, high = self._min_actions, self._max_actions
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _convert_policy_to_env_actions(self, actions):
        # stretch and translate actions from [-1, 1] range into target range
        actions = self.unscale_action(actions)

        if self._arctan2_alpha:
            action_alpha = np.arctan2(actions[0], actions[1])
        else:
            action_alpha = actions[0]

        return [action_alpha] + list(actions[2:])

    def reset(self):
        """Reset to initial position"""
        reset_pose = self._env.reset_wrap()
        return reset_pose[:self.state_dim]

    def visualize(self, ik_fail, logdir="", logfile=""):
        """
        Publish the trajectory in ROS so that it can be picked up by rviz
        Args:
            ik_fail (bool): whether to mark as failed
            logdir (str): directory to store rosbag trajectory files
            logfile (str): filename for rosbag trajectory files
        """
        vis_now = True
        path = '{}/{}'.format(logdir, logfile)
        if logfile and not os.path.exists(logdir):
            os.mkdir(logdir)
        elif not logfile:
            path = ""
        self._env.visualize(vis_now, ik_fail, path)

    def step(self, action, eval=False):
        """
        Take a step in the environment.
        Args:
            action: array of length self.action_dim with values in range [-1, 1] (automatically scaled into correct range)
            eval: whether to use the ik_fail_thresh or ik_fail_thresh_eval

        Returns:
            obs: array of length self.state_dim
            reward (float): reward
            done_return (int): whether the episode terminated, see cls.DoneReturnCode
            nr_kin_failure (int): cumulative number of kinetic failures in this episode
        """
        thres = self._ik_fail_thresh_eval if eval else self._ik_fail_thresh

        converted_actions = self._convert_policy_to_env_actions(action)
        retval = self._env.simulate_env_step_wrap(thres, self._penalty_scaling, *converted_actions)
        obs, reward, done_return, nr_kin_failures = self._parse_env_output(retval)
        return obs, reward, done_return, nr_kin_failures

    def parse_done_return(self, code):
        """
        Args:
            code (int): returned value from the env, integer in [0, 2]
        """
        return self.DoneReturnCode[code]
