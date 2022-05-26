# standard library imports
import collections
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
# library imports
import numpy as np
import torch

mpl.rcParams['agg.path.chunksize'] = 10000000

from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.optim.sgd import SGD
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# our imports
from memory import Memory
from models import Actor, DoubleCritic, SingleCritic
from utils import soft_update

# define ablations
ABLATIONS = collections.namedtuple("Ablations",
                                   "TARGET_POLICY_SMOOTHING DELAYED_POLICY_UPDATE DOUBLE_Q_LEARNING")


class TD3:
    """
    Implementation of the TD3 algorithm,
    as described in the TD3 paper.
    """

    def __init__(self,
                 env,
                 state_dim,
                 action_dim,
                 seed=None,
                 gamma=0.99,
                 tau=0.005,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 policy_delay=2,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 history_length=0,
                 lr_schedule=None,
                 ablations=ABLATIONS(TARGET_POLICY_SMOOTHING=True,
                                     DELAYED_POLICY_UPDATE=True,
                                     DOUBLE_Q_LEARNING=True),
                 device="cuda"):
        """
        Initialize the TD3 algorithm.
        Defaults for the keyword arguments
        gamma, tau, actor_lr, critic_lr,
        policy_delay, policy_noise and noise_clip
        were chosen according to the TD3 paper.
        We also use Adam as optimizer for the actor and the critic
        as suggested in the TD3 paper.
        """
        self.algorithm_name = "td3"
        # writer will be set later on in train method
        self.writer = None
        self.env = env
        self.ablations = ablations

        # set seeds
        if seed is not None:
            # self.env.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        # create a copy of the environment for evaluation, not necessary but more clean
        # self.eval_env = copy.deepcopy(self.env)

        assert history_length >= 0, "history length cannot be negative"
        # initialize the critic network and the critic target network which
        # are both double critic networks in the TD3 case (double q learning)
        # turn torch autograd off for critic target network to save computational costs
        critic_cls = DoubleCritic if ablations.DOUBLE_Q_LEARNING else SingleCritic
        self.critic = critic_cls(state_dim, action_dim, history_length)
        self.critic_target = critic_cls(state_dim, action_dim, history_length)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # initialize the actor network and the actor target network
        # turn torch autograd off for actor target network to save computational costs
        actor_cls = Actor
        self.actor = actor_cls(state_dim, action_dim, history_length)
        self.actor_target = actor_cls(state_dim, action_dim, history_length)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.gamma = gamma  # discount factor
        self.tau = tau  # update factor for updating the target networks
        # only update actor network every policy_delay steps (delayed policy update)
        self.policy_delay = policy_delay if ablations.DELAYED_POLICY_UPDATE else 1
        # policy noise added to actions by the policy (target policy smoothing)
        self.policy_noise = policy_noise
        # range from -noise_clip to noise_clip in which to clip the policy noise
        self.noise_clip = noise_clip

        self.history_length = history_length
        self.memory = Memory(state_dim, action_dim,
                             max_size=int(1e6),
                             history_length=history_length,
                             device=device)
        self.action_dim = action_dim
        self.state_dim = state_dim

        # optimizer for the critic network
        # important to optimize both of the critic networks in the double critic
        self.critic_optimizer = SGD(
            self.critic.parameters(), lr=critic_lr)

        # optimizer for the actor network
        self.actor_optimizer = SGD(
            self.actor.parameters(), lr=actor_lr)

        if lr_schedule == "step":
            self.actor_lr_scheduler = lr_scheduler.StepLR(self.actor_optimizer, 150000)
            self.critic_lr_scheduler = lr_scheduler.StepLR(self.critic_optimizer, 150000)
        elif lr_schedule == "multistep":
            self.actor_lr_scheduler = lr_scheduler.MultiStepLR(self.actor_optimizer, [400000, 800000])
            self.critic_lr_scheduler = lr_scheduler.MultiStepLR(self.critic_optimizer, [400000, 800000])
        elif lr_schedule == "cosine":
            self.actor_lr_scheduler = lr_scheduler.CosineAnnealingLR(self.actor_optimizer, 500000, 0)
            self.critic_lr_scheduler = lr_scheduler.CosineAnnealingLR(self.critic_optimizer, 500000, 0)
        elif lr_schedule == "exponential":
            self.actor_lr_scheduler = lr_scheduler.ExponentialLR(self.actor_optimizer, 0.9)
            self.critic_lr_scheduler = lr_scheduler.ExponentialLR(self.critic_optimizer, 0.9)
        else:
            self.actor_lr_scheduler = None
            self.critic_lr_scheduler = None

        self.device = torch.device(device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        self.actor.to(self.device)
        self.actor_target.to(device)

    def critic_loss(self, batch):
        """
        Compute the loss of the critic network on a batch of transitions.
        """
        s, a, r, ns, f = batch

        nf = (1 - f).view(-1, 1)

        # print(self.env.action_space.low[0], self.env.action_space.high[0])

        with torch.no_grad():
            target_a = self.actor_target(ns)

            # target policy smoothing
            if self.ablations.TARGET_POLICY_SMOOTHING:
                eps = torch.rand_like(target_a) * self.policy_noise
                eps = torch.clamp(eps, -self.noise_clip, self.noise_clip)
                target_a_smoothed = target_a + eps
                # clip action into valid range for the env
                target_a_smoothed = torch.clamp(target_a_smoothed,
                                                -1, 1)
            else:
                target_a_smoothed = target_a

            # take the minimum of the two q values we get by the critic as target
            # double q learning
            if self.ablations.DOUBLE_Q_LEARNING:
                target_q1, target_q2 = self.critic_target(ns, target_a_smoothed)
                target_q = torch.min(target_q1, target_q2)
            else:
                target_q = self.critic_target(ns, target_a_smoothed)
            target_q = r.view(-1, 1) + (nf * self.gamma * target_q)

        if self.ablations.DOUBLE_Q_LEARNING:
            # get two current q values
            q1, q2 = self.critic(s, a)

            # loss of the critic network is the sum of the MSEs between
            # corresponding current and target q values
            l1 = F.mse_loss(q1, target_q)
            l2 = F.mse_loss(q2, target_q)
            return l1 + l2
        else:
            q = self.critic(s, a)
            l = F.mse_loss(q, target_q)
            return l

    def actor_loss(self, batch):
        """
        Compute the loss of the actor network on a batch of transitions.
        """
        s, _, _, _, _ = batch

        for p in self.critic.parameters():
            p.requires_grad = False

        # we want to maximize the expectation/mean of the q values,
        # thats why we have to put a minus in front to minimize this as a loss
        a = self.actor(s)
        l = -self.critic.c1(s, a).mean()

        for p in self.critic.parameters():
            p.requires_grad = True

        return l

    def update(self, batch, update_step):
        """
        Perform a update on the actor and critic networks on a batch of transitions.
        """
        self.critic_optimizer.zero_grad()
        critic_loss = self.critic_loss(batch)
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.critic_lr_scheduler is not None:
            self.critic_lr_scheduler.step()

        # delayed policy update
        if update_step % self.policy_delay == 0:
            self.actor_optimizer.zero_grad()
            actor_loss = self.actor_loss(batch)
            actor_loss.backward()
            self.actor_optimizer.step()
            if self.actor_lr_scheduler is not None:
                self.actor_lr_scheduler.step()

            # update target networks
            with torch.no_grad():
                soft_update(self.critic_target, self.critic, self.tau)
                soft_update(self.actor_target, self.actor, self.tau)

            return critic_loss.item(), actor_loss.item()

        return critic_loss.item(), None

    def select_action(self, state, epsilon):
        """
        Select an action using the actor network while
        adding gaussian exploration noise from N(0, epsilon)
        """
        with torch.no_grad():
            a = self.actor(torch.tensor(state[np.newaxis, ...], dtype=torch.float).to(
                self.device)).cpu().numpy().squeeze()
        a += np.random.randn(self.action_dim) * epsilon
        # clip action into valid range for the env
        a = a.clip(-1, 1)
        return a

    def evaluate(self, log_dir, steps, num_episodes=30, visualize=False):
        """
        Evaluates the current state of the algorithm on 10 episodes.
        Important to use no exploration noise (epsilon=0) here.
        Tracks and returns the episode reward and episode length of the
        10 episodes.
        """
        e_rs = []
        e_ls = []
        e_fails = []
        for i in range(num_episodes):
            e_r = 0
            e_l = 0
            s = np.array(self.env.reset(), dtype=np.float32)[np.newaxis, ...]
            state_history = np.repeat(s, self.history_length + 1, axis=0) \
                if self.history_length > 0 else s
            while True:
                # use ik_fail_thresh_eval during evaluation
                s, r, done_return, nr_of_kinetic_failures = self.env.step(
                    self.select_action(state_history, 0),
                    eval=True)
                f = done_return > 0
                state_history = np.vstack([state_history, np.array(s, dtype=np.float32)])[1:, ...]
                e_r += r
                e_l += 1
                if f:
                    if visualize:
                        fail = done_return != 1
                        self.env.visualize(fail, log_dir,
                                           "{}_{}_{}_{}_{}-fails.bag".format(self.algorithm_name, steps, i,
                                                                             "success" if not fail else "fail",
                                                                             nr_of_kinetic_failures))
                    break
            e_rs.append(e_r)
            e_ls.append(e_l)
            e_fails.append(nr_of_kinetic_failures)
        return e_rs, e_ls, e_fails, float(np.where(np.array(e_fails) > self.env._ik_fail_thresh_eval, 1,
                                                   0).sum()) / float(num_episodes)

    def train(self,
              steps,
              max_episode_length,
              epsilon,
              epsilon_schedule=None,
              batch_size=100,
              initial_steps=1000,
              log_every=4000,
              save_every=20000,
              save_dir="checkpoints",
              log_dir="logs",
              tensorboard_dir="tensorboard"):
        """
        Train the TD3 algorithm for a specified number of steps.
        Defaults for the keyword arguments
        batch_size and initial_steps were chosen according to the TD3 paper.
        """
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # track time
        start_time = time.time()
        t_time = time.time()
        # track episode rewards and lengths
        e_r = 0
        e_l = 0
        episodes = 0

        s = np.array(self.env.reset(), dtype=np.float32)[np.newaxis, ...]
        state_history = np.repeat(s, self.history_length + 1, axis=0) \
            if self.history_length > 0 else s

        critic_losses = []
        actor_losses = []

        train_returns = []
        train_lengths = []
        train_fails = []

        val_returns_means = []
        val_returns_maxs = []
        val_returns_mins = []
        val_fail_percentages = []
        val_lengths_means = []
        val_lengths_maxs = []
        val_lengths_mins = []

        for t in tqdm(range(steps)):
            if epsilon_schedule is not None:
                eps = epsilon_schedule(epsilon, t, steps)
            else:
                eps = epsilon
            # sample random action for number of initial steps
            # increases exploration (see TD3 paper)
            if t > initial_steps:
                a = self.select_action(state_history, eps)
            else:
                a = np.random.uniform(-1, 1, self.env.action_dim)

            ns, r, done_return, nr_kin_failures = self.env.step(a)
            # add next state and remove oldest state
            next_state_history = np.vstack([state_history, np.array(ns, dtype=np.float32)])[1:, ...]

            f = done_return > 0  # terminal when return code is 1 or 2
            e_r += r
            e_l += 1

            self.memory.add(state_history, a, r, next_state_history, f)

            state_history = next_state_history

            if f or (e_l == max_episode_length):
                self.writer.add_scalar(
                    "{}_train_ep_return".format(self.algorithm_name), e_r, t + 1)
                self.writer.add_scalar(
                    "{}_train_ep_length".format(self.algorithm_name), e_l, t + 1)
                self.writer.add_scalar(
                    "{}_train_ep_fails".format(self.algorithm_name), nr_kin_failures, t + 1)

                s = np.array(self.env.reset(), dtype=np.float32)[np.newaxis, ...]
                state_history = np.repeat(s, self.history_length + 1, axis=0) \
                    if self.history_length > 0 else s

                e_r = 0
                e_l = 0

                episodes += 1

            # start with updating after the initial steps (see TD3 paper)
            critic_loss = 0
            last_actor_loss = 0
            if t > initial_steps:
                last_actor_loss = 0
                batch = self.memory.sample(batch_size)
                critic_loss, actor_loss = self.update(batch, t + 1)
                if actor_loss is not None:
                    last_actor_loss = actor_loss

            if ((t + 1) % save_every == 0) or (t + 1 == steps):
                self.save(save_dir, "{}_{}".format(self.algorithm_name, t + 1))

            if ((t + 1) % log_every == 0) or (t + 1 == steps):
                val_returns, val_lengths, val_fails, val_fail_percentage = self.evaluate(log_dir=log_dir,
                                                                                         steps=t + 1)
                val_returns_means.append(np.mean(val_returns))
                val_returns_maxs.append(np.max(val_returns))
                val_returns_mins.append(np.min(val_returns))
                val_lengths_means.append(np.mean(val_returns))
                val_lengths_maxs.append(np.max(val_returns))
                val_lengths_mins.append(np.min(val_returns))
                val_fail_percentages.append(val_fail_percentage)
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                train_returns.append(e_r)
                train_lengths.append(e_l)
                train_fails.append(nr_kin_failures)

                self.writer.add_scalar("{}_num_episodes".format(self.algorithm_name),
                                       episodes, t + 1)
                self.writer.add_scalar("{}_epsilon".format(self.algorithm_name),
                                       eps, t + 1)
                self.writer.add_scalar("{}_val_return_mean".format(self.algorithm_name),
                                       np.mean(val_returns), t + 1)
                self.writer.add_scalar("{}_val_return_max".format(self.algorithm_name),
                                       np.max(val_returns), t + 1)
                self.writer.add_scalar("{}_val_fail_percentage".format(self.algorithm_name),
                                       val_fail_percentage, t + 1)
                self.writer.add_histogram("{}_val_length_hist".format(self.algorithm_name),
                                          np.array(val_lengths), t + 1)
                self.writer.add_histogram("{}_val_fails_hist".format(self.algorithm_name),
                                          np.array(val_fails), t + 1)
                self.writer.add_histogram("{}_val_returns_hist".format(self.algorithm_name),
                                          np.array(val_returns), t + 1)
                self.writer.add_scalar("{}_val_return_min".format(self.algorithm_name),
                                       np.min(val_returns), t + 1)
                self.writer.add_scalar("{}_val_length_mean".format(self.algorithm_name),
                                       np.mean(val_lengths), t + 1)
                self.writer.add_scalar("{}_val_length_max".format(self.algorithm_name),
                                       np.max(val_lengths), t + 1)
                self.writer.add_scalar("{}_val_length_min".format(self.algorithm_name),
                                       np.min(val_lengths), t + 1)
                self.writer.add_scalar("{}_val_fails_mean".format(self.algorithm_name),
                                       np.mean(val_fails), t + 1)
                self.writer.add_scalar("{}_val_fails_max".format(self.algorithm_name),
                                       np.max(val_fails), t + 1)
                self.writer.add_scalar("{}_val_fails_min".format(self.algorithm_name),
                                       np.min(val_fails), t + 1)
                self.writer.add_scalar("{}_critic_loss".format(self.algorithm_name),
                                       critic_loss, t + 1)
                self.writer.add_scalar("{}_actor_loss".format(self.algorithm_name),
                                       last_actor_loss, t + 1)
                self.writer.add_scalar("{}_time_since_start".format(self.algorithm_name),
                                       time.time() - start_time, t + 1)
                self.writer.add_scalar("{}_time_since_last".format(self.algorithm_name),
                                       time.time() - t_time, t + 1)
                t_time = time.time()

        plt.figure(figsize=(20, 20))

        plt.plot(actor_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Actor losses")
        plt.savefig("Actor losses.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(critic_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Critic losses")
        plt.savefig("Critic losses.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(val_returns_means)
        plt.xlabel("Epochs")
        plt.ylabel("Return mean")
        plt.title("val_returns_means")
        plt.savefig("val_returns_means.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(val_fail_percentages)
        plt.xlabel("Epochs")
        plt.ylabel("val_fail_percentages")
        plt.title("val fail percentages")
        plt.savefig("val_fail_percentages.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(val_returns_mins)
        plt.xlabel("Epochs")
        plt.ylabel("val_returns_min")
        plt.title("val_returns_mins")
        plt.savefig("val_returns_mins.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(val_returns_means)
        plt.xlabel("Epochs")
        plt.ylabel("Return mean")
        plt.title("val_returns_means")
        plt.savefig("val_returns_means.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(val_lengths_mins)
        plt.xlabel("Epochs")
        plt.ylabel("val_lengths_min")
        plt.title("val_lengths_mins")
        plt.savefig("val_lengths_mins.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(val_lengths_maxs)
        plt.xlabel("Epochs")
        plt.ylabel("val_lengths_max")
        plt.title("val_lengths_maxs")
        plt.savefig("val_lengths_maxs.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(val_lengths_means)
        plt.xlabel("Epochs")
        plt.ylabel("val_lengths_mean")
        plt.title("val_lengths_means")
        plt.savefig("val_lengths_means.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(train_returns)
        plt.xlabel("Epochs")
        plt.ylabel("train_return")
        plt.title("train_returns")
        plt.savefig("train_returns.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(train_lengths)
        plt.xlabel("Epochs")
        plt.ylabel("train_length")
        plt.title("train_lengths")
        plt.savefig("train_lengths.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        plt.figure(figsize=(20, 20))
        plt.plot(train_fails)
        plt.xlabel("Epochs")
        plt.ylabel("train_fail")
        plt.title("train_fails")
        plt.savefig("train_fails.png")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

    def save(self, directory, filename):
        """
        Save the actor and critic as well as their optimizers to files.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, filename)
        torch.save(self.actor.state_dict(), filename + "_actor.pt")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer.pt")
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer.pt")

    def load(self, prefix_path):
        """
        Load state dicts for the actor and critic and their optimizers from files.
        """
        filename = prefix_path
        self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer.pt"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer.pt"))
