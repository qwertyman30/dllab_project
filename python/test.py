# library imports
import argparse
import json
import os
import warnings

import numpy as np

# our imports
from modulation_rl.ModulationEnv import ModulationEnv
from td3 import TD3, ABLATIONS

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='DLLab modulation task')
    parser.add_argument("checkpoint", type=str, help="checkpoint to use for testing")
    parser.add_argument("--num-episodes", type=int, help="number of evaluation episodes", default=100)
    parser.add_argument("--env", type=str,
                        default="ModulationEnv", help="set the environment")
    parser.add_argument("--algorithm", type=str, default="TD3",
                        help="set the algorithm to use")
    parser.add_argument("--random-start-pose", action="store_true",
                        help="whether to use a random starting position each episode")
    parser.add_argument("--penalty-scaling", type=float, default=0.1,
                        help="weighting factor of the penalty for large action modulations")
    parser.add_argument("--history-length", type=int, default=0,
                        help="number of past states to incorporate into the current state")
    parser.add_argument("--log-dir", type=str, default="test_logs",
                        help="set the directory for the logs")
    parser.add_argument("--seeds", type=int, nargs="+", default=42,
                        help="set seed")
    parser.add_argument("--save-all-data", action="store_true", help="whether to save all datapoints")
    parser.add_argument("--save-visualisations", action="store_true", help="whether to save rosbag files")
    parser.add_argument('--device', type=str, default="cuda",
                        help="specify the device to run the algorithm on (either cuda or cpu)")
    parser.add_argument("--no-dql", action="store_true",
                        help="whether to use double q learning")
    parser.add_argument("--no-tps", action="store_true",
                        help="whether to use target policy smoothing")
    parser.add_argument("--no-dpu", action="store_true",
                        help="whether to use delayed policy updates")
    parser.set_defaults(no_dql=False, no_tps=False, no_dpu=False, save_all=False, save_visualisations=False,
                        random_start_pose=False)
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

    steps = int(args.checkpoint.split("_")[-1])

    ablations = ABLATIONS(TARGET_POLICY_SMOOTHING=not args.no_tps,
                          DELAYED_POLICY_UPDATE=not args.no_dpu,
                          DOUBLE_Q_LEARNING=not args.no_dql)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    all_returns = all_lengths = all_fails = all_fail_percentage = None

    for seed in args.seeds:
        if args.env == "ModulationEnv":
            env = ModulationEnv(rnd_start_pose=args.random_start_pose,
                                penalty_scaling=args.penalty_scaling,
                                seed=seed,
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
                            seed=seed,
                            ablations=ablations,
                            device=args.device)
        else:
            print("Unknown algorithm {}\n".format(args.algorithm))
            exit(1)

        algorithm.load(args.checkpoint)

        print("Evaluating on {} episodes...".format(args.num_episodes))
        returns, lengths, fails, fail_percentage = algorithm.evaluate(log_dir=args.log_dir,
                                                                      steps=steps,
                                                                      num_episodes=args.num_episodes,
                                                                      visualize=args.save_visualisations)

        result_string = "Seed: {}\nReturns: {} +- {}\nFails: {} +- {}\n" \
                        "Lengths: {} +- {}\nFail percentage: {}".format(seed,
                                                                        np.mean(returns),
                                                                        np.std(returns),
                                                                        np.mean(fails),
                                                                        np.std(fails),
                                                                        np.mean(lengths),
                                                                        np.std(lengths),
                                                                        fail_percentage)
        if all_returns is None:
            all_returns = returns
            all_fails = fails
            all_fail_percentage = [fail_percentage]
            all_lengths = lengths
        all_returns.extend(returns)
        all_fails.extend(fails)
        all_fail_percentage.extend([fail_percentage])
        all_lengths.extend(lengths)

        print(result_string)

    result_string = "Seeds: {}\nReturns: {} +- {}\nFails: {} +- {}\n" \
                    "Lengths: {} +- {}\nFail percentage: {} +- {}".format(args.seeds,
                                                                          np.mean(all_returns),
                                                                          np.std(all_returns),
                                                                          np.mean(all_fails),
                                                                          np.std(all_fails),
                                                                          np.mean(all_lengths),
                                                                          np.std(all_lengths),
                                                                          np.mean(all_fail_percentage),
                                                                          np.std(all_fail_percentage))
    print(result_string)
    with open(os.path.join(args.log_dir, "results.txt"), "w") as f:
        f.write(result_string)

    if args.save_all_data:
        with open(os.path.join(args.log_dir, "results_all.json"), "w") as f:
            f.write(json.dumps({
                "seeds": args.seeds,
                "returns": all_returns,
                "lengths": all_lengths,
                "fails": all_fails,
                "fail_percentages": all_fail_percentage
            }))
