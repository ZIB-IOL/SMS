# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         main.py
# Description:  Starts up a run
# ===========================================================================

import os
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager

import torch
import wandb

from runners.ensembleRunner import ensembleRunner
from runners.pretrainedRunner import pretrainedRunner
from runners.scratchRunner import scratchRunner

from utilities.utilities import Utilities as Utils

debug = "--debug" in sys.argv
defaults = dict(
        # System
        run_id=1,                                                   # The run id, determines the original random seed
        computer=socket.gethostname(),                              # The computer that runs the experiment

        # Setup
        dataset='mnist',                                            # The dataset to use, see config.py for available options
        arch='Simple',                                              # The architecture to use, see models/ for available options
        n_epochs=2,                                                 # The number of epochs to pretrain the model for (Note: this only controls the pretraining)
        batch_size=1028,                                            # The batch size to use

        # Efficiency
        use_amp=True,                                               # Whether to use automatic mixed precision

        # Optimizer
        optimizer='SGD',                                            # The optimizer to use for pretraining/retraining, currently only SGD implemented
        learning_rate='(Linear, 0.1)',                              # The learning rate to use for pretraining
        n_epochs_warmup=None,                                       # The number of epochs to warmup the lr, must be an int
        momentum=0.9,                                               # The momentum to use for the optimizer
        weight_decay=0.0001,                                        # The weight decay to use for the optimizer

        # Sparsifying strategy
        strategy='Dense',                                           # The strategy to use, see strategies/ for available options. 'Dense' = pretraining, 'IMP' = iterative magnitude pruning, 'Ensemble' = ensembl/soup methods
        goal_sparsity=0.9,                                          # The goal sparsity to reach after n_phases many prune-retrain cycles
        pruning_selector='global',                                  # Pruning allocation, must be in ['global', 'uniform', 'random']

        # Retraining
        phase=1,                                                    # The current phase of IMP/Ensemble
        n_phases=1,                                                 # The total number of phases to run
        n_epochs_per_phase=1,                                       # The number of epochs to retrain for each phase
        retrain_schedule='LLR',                                     # The retrain lr schedule, must be in ['FT', 'LRW', 'SLR', 'CLR', 'LLR', 'ALLR']

        # Ensemble method
        ensemble_method='UniformEnsembling',                        # The ensemble/soup method to use, must be in ['UniformEnsembling', 'GreedySoup']
        ensemble_by='pruned_seed',                                  # The parameter controlling what is varied during retraining, must be in ['pruned_seed', 'weight_decay', 'retrain_length', 'retrain_schedule']
        split_val=None,                                             # The value to split the ensemble_by parameter on, e.g. ensemble_by='weight_decay' and split_val=0.0001 will retrain with a weight decay of 0.0001
        n_splits_total=2,                                           # The total number of splits we expect to have, will raise an error if more models to average found
        bn_recalibration_frac=0.2,                                  # The fraction of the dataset to use for recalibrating the batch norm layers, must be in [0,1]
    )

if not debug:
    # Set everything to None recursively
    defaults = Utils.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = Utils.update_config_with_default(config, defaults)
ngpus = torch.cuda.device_count()
if ngpus > 0:
    config.update(dict(device='cuda:0'))
else:
    config.update(dict(device='cpu'))


@contextmanager
def tempdir():
    tmp_root = '/scratch/local/'
    tmp_path = os.path.join(tmp_root, 'tmp')
    if os.path.isdir(tmp_root):
        if not os.path.isdir(tmp_path): os.mkdir(tmp_path)
        path = tempfile.mkdtemp(dir=tmp_path)
    else:
        path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:
    # At the moment, IMP is the only strategy that requires a pretrained model, all others start from scratch
    config.update({'tmp_dir': tmp_dir})

    if config.strategy == 'Ensemble':
        runner = ensembleRunner(config=config)
    elif config.strategy in 'IMP':
        # Use the pretrainedRunner
        runner = pretrainedRunner(config=config)
    elif config.strategy == 'Dense':
        # Use the scratchRunner
        runner = scratchRunner(config=config)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
