# Experiments on Emergent Communication through Negotiation

Extensions and experiments based on [Emergent Communication through Negotation](https://arxiv.org/pdf/1804.03980.pdf).
Code was initially built off of [Hugh Perkins' repo](https://github.com/ASAPPinc/emergent_comms_negotiation)

## Setup
### Requirements
I recommend running in a docker container which you can build with the `Dockerfile`.
Otherwise, use a python `venv` and install the requirements

```
python -m venv env
source env/bin/activate
pip install -r requirement.txt
```

Optionally you can also set up `PYTHONPATH` variables to facilitate imports. Assuming you cloned the repo into `$HOME/ecn` you can use

```
export PROJECT=$HOME/ecn
export PYTHONPATH=$PYTHONPATH:$PROJECT
```

### Logging
I use `wandb` for logging and recommend it! You can activate logging to it with the `--wandb` arg.

Otherwise we log in `json` format to a file in `--logdir` specified as `{args.name}_{slurmid}_{timestamp}`. `slurmid` is useful if you're running on a cluster with `slurm` and it is pulled automatically


## Run
Use the scripts in `/scripts` to run the code. Each script name corresponds to the setup it runs
- `self-*` runs selfish agents
- `*-none-*` uses no communication channels
- `*-ling-*` uses only the linguistic communication channel
- `*-prop-*` uses only the proposal communication channel
- `*-both-*` uses both the proposal and linguistic communication channels
- `*-masking` uses the masked linguistic channel, where agents learn a mask over their proposals and their linguistic utterance is simply the mask multiplied with their proposal


If you want change a flag you can add it to the end of the command, e.g. entropy regularization of term policy
```
python scripts/self-none.py --term-entropy-reg 0.1
```

The scripts define the hyperparameters for each experiment

## Reproducing Graphs
We run 5 seeds of every experiment and the plot the mean and 95% confidence interval over the seeds. Run the script with `--seed` for 5 different values (e.g. `0-4`) and log to `wandb`

From `wandb`, download the `test_reward` for all 5 seeds into a `csv` for the corresponding experiment. Plot it using `notebooks/plot_wandb.ipynb`

If you're using logfiles instead of `wandb` use `notebooks/plot_logfile.ipynb`
