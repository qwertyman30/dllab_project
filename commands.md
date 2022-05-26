## How to run this repo

Please run all commands below from the `python` subdirectory.

### Train an agent

`python train.py --steps 500000 --history-length 2 --epsilon 0.5 --epsilon-schedule exp 
--lr-schedule cosine --tensorboard-dir [dir for tensorboard logs] --save-dir [dir to save checkpoints]`

For more command line arguments see `train.py` or run `python train.py --help`.

### Test an agent

`python test.py [checkpoint] --num-episodes 100 --seeds 11 22 33 44 55 --save-all-data --save-visualisations`

The `[checkpoint]` argument has to be a prefix to a saved checkpoint. If you e.g. saved the agent checkpoints during training in 
the directory `training_checkpoints/` then a valid argument for `[checkpoint]` could be `training_checkpoints/td3_500000`, which then loads
the agent checkpoint from 500000 steps of training. 

For more command line arguments see `test.py` or run `python test.py --help`.

### Experiments and ablations

See `experiments.py` and `ablations.py` to see how the experiments and ablation study were run. They are basically 
just scripts that call `train.py` with a specific set of command line arguments.