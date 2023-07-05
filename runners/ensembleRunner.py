# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         ensembleRunner.py
# Description:  Runner class for starting from pruned models
# ===========================================================================
import itertools
import json
import math
import os
import sys
import warnings
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import wandb
from torch.cuda.amp import autocast
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from tqdm.auto import tqdm

from runners.baseRunner import baseRunner
from strategies import ensembleStrategies
from utilities.utilities import Utilities as Utils, WorstClassAccuracy, CalibrationError, Candidate


class ensembleRunner(baseRunner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k_splits_per_ensemble = None

    def find_multiple_existing_models(self, filterDict):
        """Finds existing wandb runs and downloads the model files."""
        current_phase = self.config.phase  # We are in the same phase
        filterDict['$and'].append({'config.phase': current_phase})
        filterDict['$and'].append({'config.n_splits_total': self.config.n_splits_total})
        sys.stdout.write(f"Structured pruning: {self.config.prune_structured}.\n")
        if current_phase > 1:
            # We need to specify the previous ensemble method as well
            filterDict['$and'].append({'config.ensemble_method': self.config.ensemble_method})
            filterDict['$and'].append({
                'config.split_id': self.config.split_id})  # This restricts us to stay with the same split in every phase, but this is okay
            filterDict['$and'].append({'config.k_splits_per_ensemble': self.config.k_splits_per_ensemble})

        filterDict['$and'].append({'config.ensemble_by': self.config.ensemble_by})
        filterDict['$and'].append({'config.prune_structured': self.config.prune_structured})
        entity, project = wandb.run.entity, wandb.run.project
        api = wandb.Api()
        candidate_model_list = []

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

        runs = api.runs(f"{entity}/{project}", filters=filterDict)
        runsExist = False  # If True, then there exist runs that try to set a fixed init
        for run in runs:
            if run.state != 'finished':
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
            sys.stdout.write(f"Trying to access {run.name}.\n")
            checkpoint_file = run.summary.get('final_model_file')
            try:
                if checkpoint_file is not None:
                    runsExist = True
                    sys.stdout.write(
                        f"Downloading pruned model with split {run.config['ensemble_by']} value: {run.config['split_val']}.\n")
                    run.file(checkpoint_file).download(
                        root=self.tmp_dir)
                    self.seed = run.config['seed']
                    candidate_id = (run.config['split_val'])
                    candidate_model_list.append(
                        Candidate(candidate_id=candidate_id, candidate_file=os.path.join(self.tmp_dir, checkpoint_file),
                                  candidate_run=run))
            except Exception as e:  # The run is online, but the model is not uploaded yet -> results in failing runs
                print(e)
                checkpoint_file = None
                break
        assert not (
                runsExist and checkpoint_file is None), "Runs found, but one of them has no model available -> abort."
        outputStr = f"Found {len(candidate_model_list)} pruned models with split vals {sorted([c.id for c in candidate_model_list])}" \
            if checkpoint_file is not None else "Nothing found."
        sys.stdout.write(f"Trying to find reference pruned models in project: {outputStr}\n")
        assert checkpoint_file is not None, "One of the pruned models has no model file to download, Aborting."
        assert len(candidate_model_list) == self.config.n_splits_total, "Not all pruned models were found, Aborting.\n"

        # Check whether we want to find a specific split-set
        if self.config.split_id is not None:
            sorted_split_vals = sorted(
                [c.id for c in candidate_model_list])  # Sort this to ensure deterministic order of combinations
            # Generate the set of all possible split combinations
            splitCombinations = itertools.combinations(sorted_split_vals, self.config.k_splits_per_ensemble)
            # Pick the combination with the split_id
            desired_split = list(list(splitCombinations)[self.config.split_id - 1])
            # Filter the candidate model list
            candidate_model_list = [c for c in candidate_model_list if c.id in desired_split]
            sys.stdout.write(
                f"Desired split: {desired_split} - Reduced the candidate model list to {len(candidate_model_list)} models with split vals {sorted([c.id for c in candidate_model_list])}.\n")

        return candidate_model_list

    def define_optimizer_scheduler(self):
        # Define the optimizer
        if self.config.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.)

    def transport_information(self, ref_run):
        missing_config_keys = ['momentum',
                               'n_epochs_warmup',
                               'n_epochs']  # Have to have n_epochs even though it might be specified, otherwise ALLR doesnt have this

        additional_dict = {
            'last_training_lr': ref_run.summary['final.learning_rate'],
            'final.test.accuracy': ref_run.summary['final.test']['accuracy'],
            'final.train.accuracy': ref_run.summary['final.train']['accuracy'],
            'final.train.loss': ref_run.summary['final.train']['loss'],
        }
        for key in missing_config_keys:
            if key not in self.config or self.config[key] is None:
                # Allow_val_change = true because e.g. momentum defaults to None, but shouldn't be passed here
                val = ref_run.config.get(key)  # If not found, defaults to None
                self.config.update({key: val}, allow_val_change=True)
        self.config.update(additional_dict)

        self.trained_test_accuracy = additional_dict['final.test.accuracy']
        self.trained_train_loss = additional_dict['final.train.loss']
        self.trained_train_accuracy = additional_dict['final.train.accuracy']

        # Get the wandb information about lr and fill the corresponding strategy dicts, which can then be used by rewinders
        f = ref_run.file('iteration-lr-dict.json').download(root=self.tmp_dir)
        with open(f.name) as json_file:
            loaded_dict = json.load(json_file)
            lr_dict = OrderedDict(loaded_dict)
        # Upload iteration-lr dict from self.strategy to be used during retraining
        Utils.dump_dict_to_json_wandb(dumpDict=lr_dict, name='iteration-lr-dict')

    def load_soup_model(self, ensemble_state_dict):
        # Save the ensemble state dict
        fName = f"ensemble_model.pt"
        fPath = os.path.join(self.tmp_dir, fName)
        torch.save(ensemble_state_dict, fPath)  # Save the state_dict
        self.checkpoint_file = fName

        # Actually load the model
        self.model = self.get_model(reinit=True, temporary=True)  # Load the ensembled model

    def evaluate_soup(self, data='val', ensemble_labels: torch.Tensor = None):
        # Perform an evaluation pass
        AccuracyMeter = Accuracy(num_classes=self.n_classes).to(device=self.device)
        ECEMeter = CalibrationError(norm='l1').to(device=self.device)
        MCEMeter = CalibrationError(norm='max').to(device=self.device)
        WorstClassAccuracyMeter = WorstClassAccuracy(num_classes=self.n_classes).to(device=self.device)

        if data == 'val':
            loader = self.valLoader
        elif data == 'test':
            loader = self.testLoader
        elif data == 'ood':
            loader = self.oodLoader
            if loader is None:
                sys.stdout.write(f"No OOD data found, skipping OOD evaluation.\n")
                return {}
        else:
            raise NotImplementedError

        if ensemble_labels is not None:
            sys.stdout.write(f"Performing computation of prediction ensemble {data} accuracy.\n")
        else:
            sys.stdout.write(f"Performing computation of soup {data} accuracy.\n")
        with tqdm(loader, leave=True) as pbar:
            for x_input, y_target, indices in pbar:
                # Move to CUDA if possible
                x_input = x_input.to(self.device, non_blocking=True)
                indices = indices.to(self.device, non_blocking=True)
                if ensemble_labels is not None:
                    y_target = ensemble_labels[indices]  # Avg probs/predictions of batch
                y_target = y_target.to(self.device, non_blocking=True)

                with autocast(enabled=(self.config.use_amp is True)):
                    output = self.model.train(mode=False)(x_input)
                    AccuracyMeter(output, y_target)
                    ECEMeter(output, y_target)
                    MCEMeter(output, y_target)
                    WorstClassAccuracyMeter(output, y_target)

        outputDict = {
            'accuracy': AccuracyMeter.compute().item(),
            'ece': ECEMeter.compute().item(),
            'mce': MCEMeter.compute().item(),
            'worst_class_accuracy': WorstClassAccuracyMeter.compute().item(),
        }
        return outputDict

    @torch.no_grad()
    def collect_avg_output_full(self, data: str, candidate_model_list: List[Candidate]):
        output_type = 'soft_prediction'
        assert data in ['val', 'test']
        if data == 'val':
            loader = self.valLoader
        else:
            loader = self.testLoader
        sys.stdout.write(f"\nCollecting ensemble prediction.\n")

        compute_avg_probs = (output_type in ['softmax', 'soft_prediction'])
        store_tensor = torch.zeros(len(loader.dataset), self.n_classes, device=self.device)  # On CUDA for now

        for candidate in candidate_model_list:
            # Load the candidate model
            candidate_id, candidate_file = candidate.id, candidate.file
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()

            state_dict = torch.load(candidate_file,
                                    map_location=torch.device('cpu'))
            self.load_soup_model(ensemble_state_dict=state_dict)
            with tqdm(loader, leave=True) as pbar:
                for x_input, _, indices in pbar:
                    x_input = x_input.to(self.device, non_blocking=True)  # Move to CUDA if possible
                    with autocast(enabled=(self.config.use_amp is True)):
                        output = self.model.eval()(x_input)  # Logits
                        probabilities = torch.nn.functional.softmax(output, dim=1)  # Softmax(Logits)
                        if compute_avg_probs:
                            # Just add the probabilities for the average
                            store_tensor[indices] += probabilities
                        else:
                            # Add the prediction as one hot
                            binary_tensor = torch.zeros_like(store_tensor[indices])
                            # Add the ones at corresponding entries
                            binary_tensor[torch.arange(binary_tensor.size(0)).unsqueeze(1), torch.argmax(probabilities,
                                                                                                         dim=1).unsqueeze(
                                1)] = 1.

                            store_tensor[indices] += binary_tensor

        if compute_avg_probs:
            store_tensor.mul_(1. / len(candidate_model_list))  # Weighting
        else:
            assert store_tensor.sum() == (len(candidate_model_list) * len(loader.dataset))

        if output_type in ['soft_prediction', 'hard_prediction']:
            # Take the prediction given average probabilities OR Take the most frequent prediction
            store_tensor = torch.argmax(store_tensor, dim=1)

        return store_tensor

    def run(self):
        """Function controlling the workflow of pretrainedRunner"""
        assert self.config.ensemble_by in ['pruned_seed', 'weight_decay', 'retrain_length', 'retrain_schedule']
        assert self.config.n_splits_total is not None
        assert self.config.split_val is None
        assert not (self.config.k_splits_per_ensemble is None) ^ (
                self.config.split_id is None), "Both should either be None or not None"

        if self.config.k_splits_per_ensemble is not None:
            # Compute the number of available splits as n choose k
            n = self.config.n_splits_total
            k = self.config.k_splits_per_ensemble
            assert 1 <= self.config.split_id <= math.comb(n,
                                                          k), f"Split id {self.config.split_id} > {math.comb(n, k)} is not valid, Aborting."

        # Find the reference run
        filterDict = {"$and": [{"config.run_id": self.config.run_id},
                               {"config.arch": self.config.arch},
                               {"config.optimizer": self.config.optimizer},
                               {"config.goal_sparsity": self.config.goal_sparsity},
                               {"config.n_epochs_per_phase": self.config.n_epochs_per_phase},
                               {"config.n_phases": self.config.n_phases},
                               {"config.retrain_schedule": self.config.retrain_schedule},
                               {"config.strategy": 'IMP'},
                               {"config.extended_imp": self.config.extended_imp},
                               {'config.prune_structured': self.config.prune_structured}
                               ]}

        if self.config.learning_rate is not None:
            warnings.warn(
                "You specified an explicit learning rate for retraining. Note that this only controls the selection of the pretrained model.")
            filterDict["$and"].append({"config.learning_rate": self.config.learning_rate})
        if self.config.n_epochs is not None:
            warnings.warn(
                "You specified n_epochs for retraining. Note that this only controls the selection of the pretrained model.")
            filterDict["$and"].append({"config.n_epochs": self.config.n_epochs})

        candidate_models = self.find_multiple_existing_models(filterDict=filterDict)
        wandb.config.update({'seed': self.seed})  # Push the seed to wandb

        # Set a unique random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(self.seed)  # This works if CUDA not available

        torch.backends.cudnn.benchmark = True

        self.transport_information(ref_run=candidate_models[0].run)

        self.trainLoader, self.valLoader, self.testLoader, self.trainLoader_unshuffled = self.get_dataloaders()
        self.oodLoader = self.get_ood_dataloaders()

        # We first define the ensembling strategy, create the ensemble, then use the 'Dense' strategy and regularly
        # load the model
        # Define callbacks finetuning_callback, restore_callback, save_model_callback
        callbackDict = {
            'final_log_callback': self.final_log,
            'soup_evaluation_callback': self.evaluate_soup,
            'load_soup_callback': self.load_soup_model,
            'recalibrate_bn_callback': self.recalibrate_bn,
        }
        self.ensemble_strategy = getattr(ensembleStrategies, self.config.ensemble_method)(model=None,
                                                                                          n_classes=self.n_classes,
                                                                                          config=self.config,
                                                                                          candidate_models=candidate_models,
                                                                                          runner=self,
                                                                                          callbacks=callbackDict)

        self.ensemble_strategy.collect_candidate_information()

        # Create ensemble
        ensemble_state_dict = self.ensemble_strategy.create_ensemble()

        # Save the ensemble state dict
        fName = f"ensemble_model.pt"
        fPath = os.path.join(self.tmp_dir, fName)
        torch.save(ensemble_state_dict, fPath)  # Save the state_dict
        self.checkpoint_file = fName

        # Actually load the model
        self.model = self.get_model(reinit=True, temporary=True)  # Load the ensembled model

        # Create 'Dense' as the Base Strategy
        self.strategy = self.define_strategy(use_dense_base=True)
        self.strategy.after_initialization()

        # Define optimizer to not get errors in the main evaluation (even though we do not actually use the optimizer)
        self.define_optimizer_scheduler()

        # Evaluate ensemble
        self.ensemble_strategy.final()

        self.checkpoint_file = self.save_model(model_type='ensemble')
        wandb.summary['final_model_file'] = f"ensemble_model_{self.config.ensemble_method}_{self.config.phase}.pt"
