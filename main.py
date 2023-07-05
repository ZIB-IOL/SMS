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

# Default wandb parameters
defaults = dict(
    # System
    run_id=None,
    computer=socket.gethostname(),
    # Setup
    dataset=None,
    arch=None,
    n_epochs=None,
    batch_size=None,
    # Effiency
    use_amp=True,
    # Optimizer
    optimizer=None,
    learning_rate=None,
    n_epochs_warmup=None,  # number of epochs to warmup the lr, should be an int
    momentum=None,
    weight_decay=None,
    # Sparsifying strategy
    strategy=None,
    goal_sparsity=None,
    pruning_selector=None,  # must be in ['global', 'uniform', 'random', 'LAMP']
    # Retraining
    phase=1,
    n_phases=None,  # Should be 1, except when using IMP
    n_epochs_per_phase=None,
    retrain_schedule=None,
    extended_imp=None,
    prune_structured=None,
    # Ensemble method
    ensemble_method=None,
    ensemble_by=None,
    split_val=None,
    split_count=None,
    split_id=None,
    # The specific split we want to consider, if None, all splits are considered. Otherwise there are n choose k options
    k_splits_per_ensemble=None,  # The number of splits we want to consider in an ensemble
    n_splits_total=None,  # The total number of splits we expect to have -> will raise an error if not that number
    bn_recalibration_frac=None,
)

if '--debug' in sys.argv:
    defaults.update(dict(
        # System
        run_id=1,
        computer=socket.gethostname(),
        # Setup
        dataset='mnist',
        arch='Simple',
        n_epochs=2,
        batch_size=1028,
        # Effiency
        use_amp=True,
        # Optimizer
        optimizer='SGD',
        learning_rate='(Linear, 0.1)',
        n_epochs_warmup=None,  # number of epochs to warmup the lr, should be an int
        momentum=0.9,
        weight_decay=0.0001,
        # Sparsifying strategy
        strategy='Ensemble',
        goal_sparsity=0.9,
        pruning_selector='global',  # must be in ['global', 'uniform', 'random', 'LAMP']
        # Retraining
        phase=1,
        n_phases=1,
        n_epochs_per_phase=2,
        retrain_schedule='LLR',
        extended_imp=False,
        prune_structured=False,
        # Ensemble method
        ensemble_method='UniformEnsembling',
        ensemble_by='pruned_seed',  # ['pruned_seed', 'weight_decay', 'retrain_length', 'retrain_schedule']
        split_val=None,
        split_id=None,
        # The specific split we want to consider, if None, all splits are considered. Otherwise there are n choose k options
        k_splits_per_ensemble=None,  # The number of splits we want to consider in an ensemble
        n_splits_total=3,  # The total number of splits we expect to have -> will raise an error if not that number
        bn_recalibration_frac=0.2,
    ))

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
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
