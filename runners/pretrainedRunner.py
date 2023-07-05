# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         pretrainedRunner.py
# Description:  Runner class for starting from a pretrained model
# ===========================================================================
import json
import sys
import warnings
from collections import OrderedDict

import numpy as np
import torch
import wandb

from runners.baseRunner import baseRunner
from utilities.utilities import Utilities as Utils


class pretrainedRunner(baseRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reference_run = None

    def find_existing_model(self, filterDict):
        """Finds an existing wandb run and downloads the model file."""
        phase_before_current = self.config.phase - 1
        sys.stdout.write(f"Structured pruning: {self.config.prune_structured}.\n")
        if phase_before_current > 0:
            # We specify the phase in the filterDict, because we want to find the model that was trained in the previous phase
            filterDict['$and'].append({'config.phase': phase_before_current})

            # We specify several other identifiers
            identifiers = [{"config.goal_sparsity": self.config.goal_sparsity},
                           {"config.n_epochs_per_phase": self.config.n_epochs_per_phase},
                           {"config.n_phases": self.config.n_phases},
                           {"config.retrain_schedule": self.config.retrain_schedule}]
            for identifier in identifiers:
                filterDict['$and'].append(identifier)

            sys.stdout.write(
                f"Specified ensemble_method {self.config.ensemble_method}, ensemble_by {self.config.ensemble_by}, split_val {self.config.split_val}.\n")
            filterDict['$and'].append({'config.ensemble_by': self.config.ensemble_by})
            if self.config.ensemble_method not in [None, 'None', 'none']:
                sys.stdout.write(
                    f"Looking for last ensembled model with split_id {self.config.split_id} and k_splits_per_ensemble {self.config.k_splits_per_ensemble}\n")
                filterDict['$and'].append({'config.strategy': 'Ensemble'})
                filterDict['$and'].append({'config.ensemble_method': self.config.ensemble_method})
                filterDict['$and'].append({'config.split_id': self.config.split_id})
                filterDict['$and'].append({'config.k_splits_per_ensemble': self.config.k_splits_per_ensemble})
                filterDict['$and'].append({'config.prune_structured': self.config.prune_structured})

                # We now also need to filter for n_splits_total since otherwise we use different settings
                sys.stdout.write(f"Looking for n_splits_total {self.config.n_splits_total}.\n")
                assert self.config.n_splits_total is not None
                filterDict['$and'].append({'config.n_splits_total': self.config.n_splits_total})
            else:
                # No ensemble method specified, we perform regular IMP
                sys.stdout.write("Looking for last retrained model.\n")
                filterDict['$and'].append({'config.strategy': 'IMP'})
                filterDict['$and'].append({'config.split_val': self.config.split_val})
                filterDict['$and'].append({'config.prune_structured': self.config.prune_structured})
                if self.config.extended_imp:
                    filterDict['$and'].append({'config.n_splits_total': self.config.n_splits_total})
        else:
            assert self.config.n_splits_total is not None
            filterDict['$and'].append({'config.strategy': 'Dense'})

        entity, project = wandb.run.entity, wandb.run.project
        api = wandb.Api()
        # Some variables have to be extracted from the filterDict and checked manually, e.g. weight decay in scientific format
        manualVariables = ['weight_decay', 'penalty', 'group_penalty']
        manVarDict = {}
        dropIndices = []
        for var in manualVariables:
            for i in range(len(filterDict['$and'])):
                entry = filterDict['$and'][i]
                s = f"config.{var}"
                if s in entry:
                    dropIndices.append(i)
                    manVarDict[var] = entry[s]
        for idx in reversed(sorted(dropIndices)): filterDict['$and'].pop(idx)

        checkpoint_file = None
        runs = api.runs(f"{entity}/{project}", filters=filterDict)
        runsExist = False  # If True, then there exist runs that try to set a fixed init
        for run in runs:
            if run.state == 'failed':
                # Ignore this run
                continue
            # Check if run satisfies the manual variables
            conflict = False
            for var, val in manVarDict.items():
                if var in run.config and run.config[var] != val:
                    conflict = True
                    break
            if conflict:
                continue

            checkpoint_file = run.summary.get('final_model_file')
            try:
                if checkpoint_file is not None:
                    runsExist = True
                    run.file(checkpoint_file).download(root=self.tmp_dir)
                    seed = run.config['seed']
                    reference_run = run
                    break
            except Exception as e:  # The run is online, but the model is not uploaded yet -> results in failing runs
                print(e)
                checkpoint_file = None
        assert not (
                runsExist and checkpoint_file is None), "Runs found, but none of them have a model available -> abort."
        outputStr = f"Found {checkpoint_file} in run {run.name}" \
            if checkpoint_file is not None else "Nothing found."
        sys.stdout.write(f"Trying to find reference trained model in project: {outputStr}\n")
        assert checkpoint_file is not None, "No reference trained model found, Aborting."
        return checkpoint_file, seed, reference_run

    def get_missing_config(self):
        missing_config_keys = ['momentum',
                               'n_epochs_warmup',
                               'n_epochs']  # Have to have n_epochs even though it might be specified, otherwise ALLR doesnt have this

        additional_dict = {
            'last_training_lr': self.reference_run.summary['final.learning_rate'],
            'final.test.accuracy': self.reference_run.summary['final.test']['accuracy'],
            'final.train.accuracy': self.reference_run.summary['final.train']['accuracy'],
            'final.train.loss': self.reference_run.summary['final.train']['loss'],
        }
        for key in missing_config_keys:
            if key not in self.config or self.config[key] is None:
                # Allow_val_change = true because e.g. momentum defaults to None, but shouldn't be passed here
                val = self.reference_run.config.get(key)  # If not found, defaults to None
                self.config.update({key: val}, allow_val_change=True)
        self.config.update(additional_dict)

        self.trained_test_accuracy = additional_dict['final.test.accuracy']
        self.trained_train_loss = additional_dict['final.train.loss']
        self.trained_train_accuracy = additional_dict['final.train.accuracy']

    def define_optimizer_scheduler(self):
        # Define the optimizer using the parameters from the reference run
        if self.config.optimizer == 'SGD':
            wd = self.config['weight_decay'] or 0.
            if self.config.ensemble_by == 'weight_decay':
                wd = self.config.split_val
                sys.stdout.write(f"We split by the weight decay. Value {wd}.\n")
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.config['last_training_lr'],
                                             momentum=self.config['momentum'],
                                             weight_decay=wd,
                                             nesterov=wd > 0.)

    def fill_strategy_information(self):
        # Get the wandb information about lr and fill the corresponding strategy dicts, which can then be used by rewinders
        f = self.reference_run.file('iteration-lr-dict.json').download(root=self.tmp_dir)
        with open(f.name) as json_file:
            loaded_dict = json.load(json_file)
            self.strategy.lr_dict = OrderedDict(loaded_dict)
        # Upload iteration-lr dict from self.strategy to be used during retraining
        Utils.dump_dict_to_json_wandb(dumpDict=self.strategy.lr_dict, name='iteration-lr-dict')

    def run(self):
        """Function controlling the workflow of pretrainedRunner"""
        # Find the reference run
        filterDict = {"$and": [{"config.run_id": self.config.run_id},
                               {"config.arch": self.config.arch},
                               {"config.optimizer": self.config.optimizer},
                               ]}

        assert self.config.phase is not None
        assert self.config.split_val is not None
        if self.config.ensemble_by not in [None, 'None', 'none']:
            # We do not perform regular IMP
            assert self.config.ensemble_by in ['pruned_seed', 'weight_decay', 'retrain_length', 'retrain_schedule']

        if self.config.learning_rate is not None:
            warnings.warn(
                "You specified an explicit learning rate for retraining. Note that this only controls the selection of the pretrained model.")
            filterDict["$and"].append({"config.learning_rate": self.config.learning_rate})
        if self.config.n_epochs is not None:
            warnings.warn(
                "You specified n_epochs for retraining. Note that this only controls the selection of the pretrained model.")
            filterDict["$and"].append({"config.n_epochs": self.config.n_epochs})

        if self.config.extended_imp:
            assert self.config.n_splits_total is not None, "You have to specify the total number of splits for extended IMP."

        self.checkpoint_file, self.seed, self.reference_run = self.find_existing_model(filterDict=filterDict)
        wandb.config.update({'seed': self.seed})  # Push the seed to wandb
        seed = self.seed
        if self.config.ensemble_by == 'pruned_seed':
            # We use a new seed for retraining depending on the true seed (self.seed) and the pruned_seed
            seed = self.seed + self.config.split_val
            sys.stdout.write(f"Original seed {self.seed}, new seed {seed}.\n")
        # Set a unique random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(seed)  # This works if CUDA not available

        torch.backends.cudnn.benchmark = True
        self.get_missing_config()  # Load keys that are missing in the config

        self.trainLoader, self.valLoader, self.testLoader, self.trainLoader_unshuffled = self.get_dataloaders()
        self.model = self.get_model(reinit=True, temporary=True)  # Load the previous model

        self.squared_model_norm = Utils.get_model_norm_square(model=self.model)
        # Define strategy
        self.strategy = self.define_strategy()
        self.strategy.set_to_finetuning_phase()
        self.strategy.after_initialization()  # To ensure that all parameters are properly set
        self.define_optimizer_scheduler()  # This HAS to be after the definition of the strategy, otherwise changing the models parameters will not be noticed by the optimizer!
        self.strategy.set_optimizer(opt=self.optimizer)
        self.fill_strategy_information()

        # Run the computations
        self.strategy.at_train_end()

        self.strategy.final()

        # Save pruned model, to be used by pretrainedRunner
        self.checkpoint_file = self.save_model(model_type='pruned')
        wandb.summary['final_model_file'] = f"pruned_model_{self.config.split_val}_{self.config.phase}.pt"
