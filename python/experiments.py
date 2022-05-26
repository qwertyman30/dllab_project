import os
import subprocess

if __name__ == "__main__":
    steps = 500000
    history_lengths = [0, 2, 4, 6, 8]
    epsilons = [0.5]
    epsilon_schedules = ["exp", None]
    lr_schedules = ["cosine", None]

    base_dir = "experiments"

    for hl in history_lengths:
        for eps in epsilons:
            for eps_schedule in epsilon_schedules:
                for lr_schedule in lr_schedules:
                    dirname = "hl-{}_eps-{}_epssched-{}_lrsched-{}_steps-{}".format(hl,
                                                                                    eps,
                                                                                    eps_schedule,
                                                                                    lr_schedule,
                                                                                    steps)
                    print("Running experiment: " + dirname)

                    if not os.path.exists(os.path.join(base_dir, dirname)):
                        os.makedirs(os.path.join(base_dir, dirname))
                    with open(os.path.join(base_dir, dirname, "params.csv"), "w") as f:
                        f.write("hl, {}\neps, {}\nepssched, {}\nlrsched, {}\nsteps, {}".format(hl, eps, eps_schedule,
                                                                                               lr_schedule, steps))

                    tensorboard_dir = os.path.join(base_dir, dirname, "tensorboard_" + dirname)
                    checkpoint_dir = os.path.join(base_dir, dirname, "checkpoints" + dirname)

                    command = "python train.py --steps {} --history-length {} --tensorboard-dir {} " \
                              "--save-dir {} --epsilon-schedule {} --lr-schedule {} --epsilon {}".format(steps, hl,
                                                                                                         tensorboard_dir,
                                                                                                         checkpoint_dir,
                                                                                                         eps_schedule,
                                                                                                         lr_schedule,
                                                                                                         eps)
                    subprocess.call([command], shell=True)
