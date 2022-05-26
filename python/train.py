# library imports
import argparse
import warnings

# our imports
from modulation_rl.ModulationEnv import ModulationEnv
from td3 import TD3, ABLATIONS
from utils import *

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='DLLab modulation task')
    parser.add_argument("--env", type=str,
                        default="ModulationEnv", help="set the environment")
    parser.add_argument("--random-start-pose", action="store_true",
                        help="whether to use a random starting position each episode")
    parser.add_argument("--penalty-scaling", type=float, default=0.1,
                        help="weighting factor of the penalty for large action modulations")
    parser.add_argument("--algorithm", type=str, default="TD3",
                        help="set the algorithm to use")
    parser.add_argument('--device', type=str, default="cuda",
                        help="specify the device to run the algorithm on (either cuda or cpu)")
    parser.add_argument('--epsilon', type=float, default=0.9,
                        help="set the exploration rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="set seed")
    parser.add_argument("--steps", type=int, default=int(1e6),
                        help="set number of steps")
    parser.add_argument("--history-length", type=int, default=0,
                        help="number of past states to incorporate into the current state")
    parser.add_argument("--max-episode-length", type=int, default=1000,
                        help="maximum number of time steps per episode")
    parser.add_argument("--initial-steps", type=int, default=1000,
                        help="number of steps to run with random actions before training")
    parser.add_argument("--log-every", type=int, default=5000,
                        help="log every n steps")
    parser.add_argument("--save-every", type=int, default=50000,
                        help="save algorithm state every n steps")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="set the batch size")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="set the directory for the logs")
    parser.add_argument("--tensorboard-dir", type=str, default="tensorboard",
                        help="set the directory for the tensorboard logs")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="set the directory for the saving of the algorithm checkpoints")
    parser.add_argument("--epsilon-schedule", type=str, default=None,
                        help="use default or custom epsilon schedule")
    parser.add_argument("--lr-schedule", type=str, default=None,
                        help="use a learning rate scheduler")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="set the location of a trained network you want to continue training")
    parser.add_argument("--no-dql", action="store_true",
                        help="whether to use double q learning")
    parser.add_argument("--no-tps", action="store_true",
                        help="whether to use target policy smoothing")
    parser.add_argument("--no-dpu", action="store_true",
                        help="whether to use delayed policy updates")
    parser.set_defaults(no_dql=False, no_tps=False, no_dpu=False, random_start_pose=False)
    args = parser.parse_args()
    return args


class Config:
    def __init__(self):
        pass

    ik_fail_thresh = 100
    ik_fail_thresh_eval = 100
    # feel free to also directly adjust the form of the reward function in the c++ code
    # only suggestion is to keep it simple. Too much reward engineering does not generalise well
    penalty_scaling = 0.1
    # use random start pose to increase difficulty (but also ability to generalise)
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


if __name__ == "__main__":
    args = parse_args()

    if args.epsilon_schedule == "exp":
        schedule = exp_decay
    elif args.epsilon_schedule == "linear":
        schedule = linear_decay
    else:
        schedule = None

    ablations = ABLATIONS(TARGET_POLICY_SMOOTHING=not args.no_tps,
                          DELAYED_POLICY_UPDATE=not args.no_dpu,
                          DOUBLE_Q_LEARNING=not args.no_dql)

    if args.env == "ModulationEnv":
        env = ModulationEnv(rnd_start_pose=args.random_start_pose,
                            penalty_scaling=args.penalty_scaling,
                            seed=args.seed,
                            ik_fail_thresh=Config.ik_fail_thresh,
                            ik_fail_thresh_eval=Config.ik_fail_thresh_eval,
                            arctan2_alpha=Config.arctan2_alpha,
                            min_actions=Config.min_action,
                            max_actions=Config.max_action)
    else:
        print("Unkown environment {}\n".format(args.env))
        exit(1)

    state_dim = env.state_dim
    action_dim = env.action_dim

    if args.algorithm == "TD3":
        algorithm = TD3(env,
                        state_dim,
                        action_dim,
                        history_length=args.history_length,
                        seed=args.seed,
                        device=args.device,
                        lr_schedule=args.lr_schedule,
                        ablations=ablations)
    else:
        print("Unknown algorithm {}\n".format(args.algorithm))
        exit(1)

    if args.warm_start is not None:
        algorithm.load(args.warm_start)
        print("warm starting networks from {}\n".format(args.warm_start))

    algorithm.train(steps=args.steps,
                    max_episode_length=args.max_episode_length,
                    epsilon=args.epsilon,
                    initial_steps=args.initial_steps,
                    epsilon_schedule=schedule,
                    batch_size=args.batch_size,
                    log_every=args.log_every,
                    save_every=args.save_every,
                    log_dir=args.log_dir,
                    tensorboard_dir=args.tensorboard_dir,
                    save_dir=args.save_dir)
