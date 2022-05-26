import os
import subprocess

if __name__ == "__main__":
    steps = 500000
    seeds = [77]
    double_q_learning = [True, False]
    target_policy_smoothing = [True, False]
    delayed_policy_update = [True, False]

    base_dir = "experiments/ablations"

    for seed in seeds:
        for x in double_q_learning:
            for y in target_policy_smoothing:
                for z in delayed_policy_update:
                    dirname = "dql-{}_tps-{}_dpu-{}_steps-{}_seed-{}".format(x, y, z,
                                                                             steps, seed)
                    print("Running experiment: " + dirname)

                    if not os.path.exists(os.path.join(base_dir, dirname)):
                        os.makedirs(os.path.join(base_dir, dirname))
                    with open(os.path.join(base_dir, dirname, "params.csv"), "w") as f:
                        f.write("dql, {}\ntps, {}\ndpu, {}".format(x, y, z))

                    tensorboard_dir = os.path.join(base_dir, dirname, "tensorboard_" + dirname)
                    checkpoint_dir = os.path.join(base_dir, dirname, "checkpoints" + dirname)

                    command = "python train.py --steps {} --seed {} --history-length 2 --tensorboard-dir {} " \
                              "--save-dir {} --epsilon-schedule exp --lr-schedule cosine --epsilon 0.5 " \
                              "{} {} {}".format(steps, seed, tensorboard_dir, checkpoint_dir, "" if x else "--no-dql",
                                                "" if y else "--no-tps", "" if z else "--no-dpu")
                    subprocess.call([command], shell=True)
