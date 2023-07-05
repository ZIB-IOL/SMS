# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         baseRunner.py
# Description:  Base Runner class, all other runners inherit from this one
# ===========================================================================
import importlib
import os
import sys
import time
from collections import OrderedDict
from math import sqrt

import numpy as np
import torch
import torch.nn.utils.prune as prune
import wandb
from torch.cuda.amp import autocast
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy as Accuracy
from tqdm.auto import tqdm

from config import datasetDict, trainTransformDict, testTransformDict
from metrics import metrics
from strategies import strategies as usual_strategies
from utilities.lr_schedulers import SequentialSchedulers, FixedLR
from utilities.utilities import Utilities as Utils
from utilities.utilities import WorstClassAccuracy, CalibrationError


class baseRunner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config):

        self.config = config
        self.dataParallel = (torch.cuda.device_count() > 1)
        if not self.dataParallel:
            self.device = torch.device(config.device)
            if 'gpu' in config.device:
                torch.cuda.set_device(self.device)
        else:
            # Use all visible GPUs
            self.device = torch.device("cuda:0")
            torch.cuda.device(self.device)

        # Set a couple useful variables
        self.checkpoint_file = None
        self.trained_test_accuracy = None
        self.trained_train_loss = None
        self.trained_train_accuracy = None
        self.after_pruning_metrics = None
        self.seed = None
        self.squared_model_norm = None
        self.n_warmup_epochs = None
        self.trainIterationCtr = 1
        self.tmp_dir = config['tmp_dir']
        sys.stdout.write(f"Using temporary directory {self.tmp_dir}.\n")
        self.ampGradScaler = None  # Note: this must be reset before training, and before retraining
        self.num_workers = None

        # Variables to be set by inheriting classes
        self.strategy = None
        self.ensemble_strategy = None
        self.trainLoader = None
        self.valLoader = None
        self.testLoader = None
        self.trainLoader_unshuffled = None
        self.oodLoader = None
        self.n_datapoints = None
        self.model = None
        self.dense_model = None
        self.wd_scheduler = None
        self.trainData = None
        self.n_total_iterations = None

        self.ultimate_log_dict = None

        if self.config.dataset in ['mnist', 'cifar10']:
            self.n_classes = 10
        elif self.config.dataset in ['cifar100']:
            self.n_classes = 100
        elif self.config.dataset in ['tinyimagenet']:
            self.n_classes = 200
        elif self.config.dataset in ['imagenet']:
            self.n_classes = 1000
        else:
            raise NotImplementedError

        # Define the loss object and metrics
        # Important note: for the correct computation of loss/accuracy it's important to have reduction == 'mean'
        self.loss_criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device=self.device)

        self.metrics = {mode: {'loss': MeanMetric().to(device=self.device),
                               'accuracy': Accuracy(num_classes=self.n_classes).to(device=self.device),
                               'ips_throughput': MeanMetric().to(device=self.device)}
                        for mode in ['train', 'val', 'test', 'ood']}
        for mode in ['val', 'test', 'ood']:
            self.metrics[mode]['ece'] = CalibrationError(norm='l1').to(device=self.device)
            self.metrics[mode]['mce'] = CalibrationError(norm='max').to(device=self.device)
            self.metrics[mode]['worst_class_accuracy'] = WorstClassAccuracy(num_classes=self.n_classes).to(
                device=self.device)

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

    def get_metrics(self):
        with torch.no_grad():
            n_total, n_nonzero = metrics.get_parameter_count(model=self.model)

            x_input, y_target, indices = next(iter(self.valLoader))
            x_input, y_target = x_input.to(self.device), y_target.to(self.device)  # Move to CUDA if possible
            n_flops, n_nonzero_flops = metrics.get_flops(model=self.model, x_input=x_input)

            distance_to_pruned, rel_distance_to_pruned = {}, {}
            if self.config.goal_sparsity is not None:
                distance_to_pruned, rel_distance_to_pruned = metrics.get_distance_to_pruned(model=self.model,
                                                                                            sparsity=self.config.goal_sparsity)

            soup_metrics = self.ensemble_strategy.get_ensemble_metrics() if self.ensemble_strategy is not None else {}
            loggingDict = dict(
                train={metric_name: metric.compute() for metric_name, metric in self.metrics['train'].items() if
                       getattr(metric, 'mode', True) is not None},  # Check if metric computable
                val={metric_name: metric.compute() for metric_name, metric in self.metrics['val'].items()},
                global_sparsity=metrics.global_sparsity(module=self.model),
                modular_sparsity=metrics.modular_sparsity(parameters_to_prune=self.strategy.parameters_to_prune),
                n_total_params=n_total,
                n_nonzero_params=n_nonzero,
                nonzero_inference_flops=n_nonzero_flops,
                baseline_inference_flops=n_flops,
                theoretical_speedup=metrics.get_theoretical_speedup(n_flops=n_flops, n_nonzero_flops=n_nonzero_flops),
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
                distance_to_origin=metrics.get_distance_to_origin(self.model),
                distance_to_pruned=distance_to_pruned,
                rel_distance_to_pruned=rel_distance_to_pruned,
                soup_metrics=soup_metrics,
            )

            for split in ['test', 'ood']:
                loggingDict[split] = dict()
                for metric_name, metric in self.metrics[split].items():
                    try:
                        # Catch case where MeanMetric mode not set yet
                        loggingDict[split][metric_name] = metric.compute()
                    except Exception as e:
                        continue

        return loggingDict

    def get_dataset_root(self, dataset_name: str) -> str:
        """Copies the dataset and returns the rootpath."""
        # Determine where the data lies
        for root in ['/software/pytorch_datasets/', './datasets_pytorch/']:
            rootPath = f"{root}{dataset_name}"
            if os.path.isdir(rootPath):
                break

        return rootPath

    def get_ood_dataloaders(self):
        if self.config.dataset == 'cifar10':
            ood_dataset_name = 'CIFAR10CORRUPT'
        elif self.config.dataset == 'cifar100':
            ood_dataset_name = 'CIFAR100CORRUPT'
        else:
            return None

        sys.stdout.write(f"Loading {ood_dataset_name} dataset for OOD performance.\n")
        ood_root = self.get_dataset_root(ood_dataset_name)
        ood_dataset = Utils.get_overloaded_dataset(datasetDict[ood_dataset_name])(root=ood_root,
                                                                                  transform=testTransformDict[
                                                                                      self.config.dataset])
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=self.config.batch_size, shuffle=False,
                                                 pin_memory=torch.cuda.is_available(), num_workers=self.num_workers)

        return ood_loader

    def get_dataloaders(self):
        rootPath = self.get_dataset_root(dataset_name=self.config.dataset)

        if self.config.dataset in ['imagenet']:
            trainData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(root=rootPath, split='train',
                                                                                       transform=trainTransformDict[
                                                                                           self.config.dataset])
            testData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(root=rootPath, split='val',
                                                                                      transform=testTransformDict[
                                                                                          self.config.dataset])
        elif self.config.dataset == 'tinyimagenet':
            traindir = os.path.join(rootPath, 'train')
            valdir = os.path.join(rootPath, 'val')
            trainData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(root=traindir,
                                                                                       transform=trainTransformDict[
                                                                                           self.config.dataset])
            testData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(root=valdir,
                                                                                      transform=testTransformDict[
                                                                                          self.config.dataset])
        else:
            trainData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(root=rootPath, train=True,
                                                                                       download=True,
                                                                                       transform=trainTransformDict[
                                                                                           self.config.dataset])

            testData = Utils.get_overloaded_dataset(datasetDict[self.config.dataset])(root=rootPath, train=False,
                                                                                      transform=testTransformDict[
                                                                                          self.config.dataset])
        train_size = int(0.9 * len(trainData))
        val_size = len(trainData) - train_size
        self.trainData, valData = torch.utils.data.random_split(trainData, [train_size, val_size],
                                                                generator=torch.Generator().manual_seed(42))
        self.n_datapoints = train_size

        if self.config.dataset in ['imagenet', 'cifar100', 'tinyimagenet']:
            self.num_workers = 4 * torch.cuda.device_count() if torch.cuda.is_available() else 0
        else:
            self.num_workers = 2 if torch.cuda.is_available() else 0

        trainLoader = torch.utils.data.DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                                  pin_memory=torch.cuda.is_available(), num_workers=self.num_workers)
        trainLoader_unshuffled = torch.utils.data.DataLoader(self.trainData, batch_size=self.config.batch_size,
                                                             shuffle=False,
                                                             pin_memory=torch.cuda.is_available(),
                                                             num_workers=self.num_workers)
        valLoader = torch.utils.data.DataLoader(valData, batch_size=self.config.batch_size, shuffle=False,
                                                pin_memory=torch.cuda.is_available(), num_workers=self.num_workers)
        testLoader = torch.utils.data.DataLoader(testData, batch_size=self.config.batch_size, shuffle=False,
                                                 pin_memory=torch.cuda.is_available(), num_workers=self.num_workers)

        return trainLoader, valLoader, testLoader, trainLoader_unshuffled

    def get_model(self, reinit: bool, temporary: bool = True) -> torch.nn.Module:
        if reinit:
            # Define the model
            model = getattr(importlib.import_module('models.' + self.config.dataset), self.config.arch)()
        else:
            # The model has been initialized already
            model = self.model

        file = self.checkpoint_file
        masks = None
        if file is not None:
            dir = wandb.run.dir if not temporary else self.tmp_dir
            fPath = os.path.join(dir, file)

            state_dict = torch.load(fPath, map_location=self.device)

            new_state_dict = OrderedDict()
            masks = OrderedDict()
            mask_module_names = []
            require_DP_format = isinstance(model,
                                           torch.nn.DataParallel)  # If true, ensure all keys start with "module."
            for k, v in state_dict.items():
                is_in_DP_format = k.startswith("module.")
                if require_DP_format and is_in_DP_format:
                    name = k
                elif require_DP_format and not is_in_DP_format:
                    name = "module." + k  # Add 'module' prefix
                elif not require_DP_format and is_in_DP_format:
                    name = k[7:]  # Remove 'module.'
                elif not require_DP_format and not is_in_DP_format:
                    name = k

                v_new = v  # Remains unchanged if not in _orig format
                if k.endswith("_orig"):
                    # We loaded the _orig tensor and corresponding mask
                    name = name[:-5]  # Truncate the "_orig"
                    if f"{k[:-5]}_mask" in state_dict.keys():
                        # Split name into the modules name and the param_type (i.e. weight, bias or similar)
                        module_name, param_type = name.rsplit(".", 1)

                        masks[(module_name, param_type)] = state_dict[f"{k[:-5]}_mask"]
                        mask_module_names.append(module_name)

                new_state_dict[name] = v_new

            maskKeys = [k for k in new_state_dict.keys() if k.endswith("_mask")]
            for k in maskKeys:
                del new_state_dict[k]

            # Load the state_dict
            model.load_state_dict(new_state_dict)

            module_dict = {}
            for name, module in model.named_modules():
                if name in mask_module_names:
                    module_dict[name] = module

        if self.dataParallel and reinit and not isinstance(model,
                                                           torch.nn.DataParallel):  # Only apply DataParallel when re-initializing the model!
            # We use DataParallelism
            model = torch.nn.DataParallel(model)
        model = model.to(device=self.device)

        # We reinforce the previous pruning
        if masks is not None:
            for (module_name, param_type), v in masks.items():
                module = module_dict[module_name]
                v = v.to(self.device)
                prune.custom_from_mask(module, name=param_type, mask=v)

        return model

    def define_optimizer_scheduler(self):
        # Learning rate scheduler in the form (type, kwargs)
        tupleStr = self.config.learning_rate.strip()
        # Remove parenthesis
        if tupleStr[0] == '(':
            tupleStr = tupleStr[1:]
        if tupleStr[-1] == ')':
            tupleStr = tupleStr[:-1]
        name, *kwargs = tupleStr.split(',')
        if name in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'Linear', 'Cosine', 'Constant']:
            scheduler = (name, kwargs)
            self.initial_lr = float(kwargs[0])
        else:
            raise NotImplementedError(f"LR Scheduler {name} not implemented.")

        # Define the optimizer
        if self.config.optimizer == 'SGD':
            wd = self.config['weight_decay'] or 0.
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.initial_lr,
                                             momentum=self.config.momentum,
                                             weight_decay=wd, nesterov=wd > 0.)

        # We define a scheduler. All schedulers work on a per-iteration basis
        iterations_per_epoch = len(self.trainLoader)
        n_total_iterations = iterations_per_epoch * self.config.n_epochs
        self.n_total_iterations = n_total_iterations
        n_warmup_iterations = 0

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups: param_group['lr'] = self.initial_lr

        # Define the warmup scheduler if needed
        warmup_scheduler, milestone = None, None
        if self.config.n_epochs_warmup and self.config.n_epochs_warmup > 0:
            assert int(
                self.config.n_epochs_warmup) == self.config.n_epochs_warmup, "At the moment no float warmup allowed."
            n_warmup_iterations = int(float(self.config.n_epochs_warmup) * iterations_per_epoch)
            # As a start factor we use 1e-20, to avoid division by zero when putting 0.
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1e-20, end_factor=1.,
                                                                 total_iters=n_warmup_iterations)
            milestone = n_warmup_iterations + 1

        n_remaining_iterations = n_total_iterations - n_warmup_iterations

        name, kwargs = scheduler
        scheduler = None
        if name == 'Constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer,
                                                            factor=1.0,
                                                            total_iters=n_remaining_iterations)
        elif name == 'StepLR':
            # Tuple of form ('StepLR', initial_lr, step_size, gamma)
            # Reduces initial_lr by gamma every step_size epochs
            step_size, gamma = int(kwargs[1]), float(kwargs[2])

            # Convert to iterations
            step_size = iterations_per_epoch * step_size

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size,
                                                        gamma=gamma)
        elif name == 'MultiStepLR':
            # Tuple of form ('MultiStepLR', initial_lr, milestones, gamma)
            # Reduces initial_lr by gamma every epoch that is in the list milestones
            milestones, gamma = kwargs[1].strip(), float(kwargs[2])
            # Remove square bracket
            if milestones[0] == '[':
                milestones = milestones[1:]
            if milestones[-1] == ']':
                milestones = milestones[:-1]
            # Convert to iterations directly
            milestones = [int(ms) * iterations_per_epoch for ms in milestones.split('|')]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=milestones,
                                                             gamma=gamma)
        elif name == 'ExponentialLR':
            # Tuple of form ('ExponentialLR', initial_lr, gamma)
            gamma = float(kwargs[1])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma)
        elif name == 'Linear':
            if len(kwargs) == 2:
                # The final learning rate has also been passed
                end_factor = float(kwargs[1]) / float(kwargs[0])
            else:
                end_factor = 0.
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                          start_factor=1.0, end_factor=end_factor,
                                                          total_iters=n_remaining_iterations)
        elif name == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                   T_max=n_remaining_iterations, eta_min=0.)

        # Reset base lrs to make this work
        scheduler.base_lrs = [self.initial_lr if warmup_scheduler else 0. for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        elif name in ['StepLR', 'MultiStepLR']:
            # We need parallel schedulers, since the steps should be counted during warmup
            self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[warmup_scheduler, scheduler])
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    def define_strategy(self, use_dense_base=False):
        #### UNSTRUCTURED
        # Define callbacksfinetuning_callback, restore_callback, save_model_callback
        callbackDict = {
            'after_pruning_callback': self.after_pruning_callback,
            'finetuning_callback': self.fine_tuning,
            'restore_callback': self.restore_model,
            'save_model_callback': self.save_model,
            'final_log_callback': self.final_log,
        }
        # Base strategies
        if use_dense_base:
            return getattr(usual_strategies, 'Dense')(model=self.model, n_classes=self.n_classes,
                                                      config=self.config, callbacks=callbackDict)
        else:
            return getattr(usual_strategies, self.config.strategy)(model=self.model, n_classes=self.n_classes,
                                                                   config=self.config, callbacks=callbackDict)

    def log(self, runTime, finetuning: bool = False, final_logging: bool = False):
        loggingDict = self.get_metrics()
        loggingDict.update({'epoch_run_time': runTime})
        if not finetuning:
            # Update final trained metrics (necessary to be able to filter via wandb)
            for metric_type, val in loggingDict.items():
                wandb.run.summary[f"final.{metric_type}"] = val
            # The usual logging of one epoch
            wandb.log(
                loggingDict
            )

        else:
            if not final_logging:
                wandb.log(
                    dict(finetune=loggingDict,
                         ),
                )
            else:
                # We add the after_pruning_metrics and don't commit, since the values are updated by self.final_log
                self.ultimate_log_dict = dict(finetune=loggingDict,
                                              pruned=self.after_pruning_metrics,
                                              )

    def final_log(self):
        """This function can ONLY be called by pretrained strategies using the final sparsified model"""
        # Recompute accuracy and loss
        sys.stdout.write(
            f"\nFinal logging\n")
        self.reset_averaged_metrics()
        if self.config.strategy != 'Dense':
            # We recalibrate the BN statistics also for IMP
            self.recalibrate_bn()
        self.evaluate_model(data='val')
        self.evaluate_model(data='test')
        self.evaluate_model(data='ood')

        # Update final trained metrics (necessary to be able to filter via wandb)
        loggingDict = self.get_metrics()
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"final.{metric_type}"] = val

        # Update after prune metrics
        if self.after_pruning_metrics is not None:
            for metric_type, val in self.after_pruning_metrics.items():
                wandb.run.summary[f"pruned.{metric_type}"] = val

        # Add to existing self.ultimate_log_dict which was not commited yet
        if self.ultimate_log_dict is not None:
            if loggingDict['train']['accuracy'] == 0:
                # we did not perform the recomputation, use the old values for train
                del loggingDict['train']

            self.ultimate_log_dict['finetune'].update(loggingDict)
        else:
            self.ultimate_log_dict = {'finetune': loggingDict}

        wandb.log(self.ultimate_log_dict)
        Utils.dump_dict_to_json_wandb(metrics.per_layer_sparsity(model=self.model), 'sparsity_distribution')

    def after_pruning_callback(self):
        """Collects pruning metrics. Is called ONCE per run, namely on the LAST PRUNING step."""

        # Make the pruning permanent (this is in conflict with strategies that do not have a permanent pruning)
        self.strategy.enforce_prunedness()

        # Compute losses, accuracies after pruning
        sys.stdout.write(f"\nGoal sparsity reached - Computing incurred losses after pruning.\n")
        self.reset_averaged_metrics()

        # self.evaluate_model(data='train')
        self.evaluate_model(data='val')
        self.evaluate_model(data='test')
        if self.squared_model_norm is not None:
            L2_norm_square = Utils.get_model_norm_square(self.model)
            norm_drop = sqrt(abs(self.squared_model_norm - L2_norm_square))
            if float(sqrt(self.squared_model_norm)) > 0:
                relative_norm_drop = norm_drop / float(sqrt(self.squared_model_norm))
            else:
                relative_norm_drop = {}
        else:
            norm_drop, relative_norm_drop = {}, {}

        pruning_instability, pruning_stability = {}, {}
        if self.trained_test_accuracy is not None and self.trained_test_accuracy > 0:
            pruning_instability = (
                                          self.trained_test_accuracy - self.metrics['test'][
                                      'accuracy'].compute()) / self.trained_test_accuracy
            pruning_stability = 1 - pruning_instability

        self.after_pruning_metrics = dict(
            val={metric_name: metric.compute() for metric_name, metric in self.metrics['val'].items()},
            test={metric_name: metric.compute() for metric_name, metric in self.metrics['test'].items()},
            norm_drop=norm_drop,
            relative_norm_drop=relative_norm_drop,
            pruning_instability=pruning_instability,
            pruning_stability=pruning_stability,
        )

        # Reset squared model norm for following pruning steps, otherwise ALLR does not work properly
        self.squared_model_norm = Utils.get_model_norm_square(model=self.model)

    def restore_model(self) -> None:
        sys.stdout.write(
            f"Restoring model from {self.checkpoint_file}.\n")
        self.model = self.get_model(reinit=False, temporary=True)

    def save_model(self, model_type: str, remove_pruning_hooks: bool = False, temporary: bool = False) -> str:
        if model_type not in ['initial', 'trained', 'pruned', 'ensemble']:
            print(f"Ignoring to save {model_type} for now.")
            return None
        fName = f"{model_type}_model.pt"
        if model_type == 'pruned':
            fName = f"{model_type}_model_{self.config.split_val}_{self.config.phase}.pt"
        elif model_type == 'ensemble':
            fName = f"{model_type}_model_{self.config.ensemble_method}_{self.config.phase}.pt"
        fPath = os.path.join(wandb.run.dir, fName) if not temporary else os.path.join(self.tmp_dir, fName)
        if remove_pruning_hooks:
            self.strategy.make_pruning_permanent(model=self.model)

        # Only save models in their non-module version, to avoid problems when loading
        try:
            model_state_dict = self.model.module.state_dict()
        except AttributeError:
            model_state_dict = self.model.state_dict()

        torch.save(model_state_dict, fPath)  # Save the state_dict
        return fPath

    def evaluate_model(self, data='train'):
        return self.train_epoch(data=data, is_training=False)

    def define_retrain_schedule(self, n_epochs_finetune, pruning_sparsity):
        """Define the retraining schedule.
            - Tuneable schedules all require both an initial value as well as a warmup length
            - Fixed schedules require no additional parameters and are mere conversions such as LRW
        """
        fixed_schedules = ['FT',  # Use last lr of original training as schedule (Han et al.), no warmup
                           'LRW',  # Learning Rate Rewinding (Renda et al.), no warmup
                           'SLR',  # Scaled Learning Rate Restarting (Le et al.), maxLR init, 10% warmup
                           'CLR',  # Cyclic Learning Rate Restarting (Le et al.), maxLR init, 10% warmup
                           'LLR',  # Linear from the largest original lr to 0, maxLR init, 10% warmup
                           'ALLR',  # LLR, but choose initial value adaptively
                           ]
        retrain_schedule = self.config.retrain_schedule
        init_val = None
        if self.config.ensemble_by == 'retrain_schedule':
            retrain_schedule = self.config.split_val
            # Check if the retrain schedule is a float
            init_val = float(retrain_schedule)
            retrain_schedule = 'LLR'
            sys.stdout.write(f"We split by the retrain schedule initial value. Value {init_val}.\n")

        # Define the initial lr, max lr and min lr
        maxLR = max(
            self.strategy.lr_dict.values())
        after_warmup_index = (self.config.n_epochs_warmup or 0) * len(self.trainLoader)
        minLR = min(list(self.strategy.lr_dict.values())[after_warmup_index:])  # Ignores warmup in orig. schedule

        n_total_iterations = len(self.trainLoader) * n_epochs_finetune

        if retrain_schedule in fixed_schedules:
            # Define warmup length
            if retrain_schedule in ['FT', 'LRW']:
                n_warmup_iterations = 0
            else:
                # 10% warmup
                n_warmup_iterations = int(0.1 * n_total_iterations)

            # Define the after_warmup_lr
            if init_val is not None:
                after_warmup_lr = init_val
            elif retrain_schedule == 'FT':
                after_warmup_lr = minLR
            elif retrain_schedule == 'LRW':
                after_warmup_lr = list(self.strategy.lr_dict.values())[
                    -n_total_iterations]  # == remaining iterations since we don't do warmup
            elif retrain_schedule in ['ALLR']:
                minLRThreshold = min(float(n_epochs_finetune) / self.config.n_epochs, 1.0) * maxLR
                # Use the norm drop
                relative_norm_drop = self.after_pruning_metrics['relative_norm_drop']
                scaling = relative_norm_drop / sqrt(pruning_sparsity)

                discounted_LR = float(scaling) * maxLR

                after_warmup_lr = np.clip(discounted_LR, a_min=minLRThreshold, a_max=maxLR)

            elif retrain_schedule in ['SLR', 'CLR', 'LLR']:
                after_warmup_lr = maxLR
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Set the optimizer lr
        for param_group in self.optimizer.param_groups:
            if n_warmup_iterations > 0:
                # If warmup, then we actually begin with 0 and increase to after_warmup_lr
                param_group['lr'] = 0.0
            else:
                param_group['lr'] = after_warmup_lr

        # Define warmup scheduler
        warmup_scheduler, milestone = None, None
        if n_warmup_iterations > 0:
            warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR \
                (self.optimizer, T_max=n_warmup_iterations, eta_min=after_warmup_lr)
            milestone = n_warmup_iterations + 1

        # Define scheduler after the warmup
        n_remaining_iterations = n_total_iterations - n_warmup_iterations
        scheduler = None
        if retrain_schedule in ['FT']:
            # Does essentially nothing but keeping the smallest learning rate
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer=self.optimizer,
                                                            factor=1.0,
                                                            total_iters=n_remaining_iterations)
        elif retrain_schedule == 'LRW':
            iterationsLR = list(self.strategy.lr_dict.values())[(-n_remaining_iterations):]
            iterationsLR.append(iterationsLR[-1])  # Double the last learning rate so we avoid the IndexError
            scheduler = FixedLR(optimizer=self.optimizer, lrList=iterationsLR)

        elif retrain_schedule in ['SLR']:
            iterationsLR = [lr if int(it) >= after_warmup_index else maxLR
                            for it, lr in self.strategy.lr_dict.items()]

            interpolation_width = (len(self.strategy.lr_dict)) / n_remaining_iterations  # In general not an integer
            reducedLRs = [iterationsLR[int(j * interpolation_width)] for j in range(n_remaining_iterations)]
            # Add a last LR to avoid IndexError
            reducedLRs = reducedLRs + [reducedLRs[-1]]

            lr_lambda = lambda it: reducedLRs[it] / float(maxLR)  # Function returning the correct learning rate factor
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        elif retrain_schedule in ['CLR']:
            stopLR = minLR
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR \
                (self.optimizer, T_max=n_remaining_iterations, eta_min=stopLR)

        elif retrain_schedule in ['LLR', 'ALLR']:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                          start_factor=1.0, end_factor=0.,
                                                          total_iters=n_remaining_iterations)

        # Reset base lrs to make this work
        scheduler.base_lrs = [after_warmup_lr for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    def fine_tuning(self, pruning_sparsity, n_epochs_finetune, phase=1):
        if n_epochs_finetune == 0:
            return
        if self.config.ensemble_by == 'retrain_length':
            n_epochs_finetune = self.config.split_val
            sys.stdout.write(f"We split by the retrain length. Value {n_epochs_finetune}.\n")
        elif self.config.extended_imp:
            n_epochs_finetune = self.config.n_splits_total * n_epochs_finetune
            sys.stdout.write(
                f"Extended IMP is enabled. We will retrain {self.config.n_splits_total} times as long: {n_epochs_finetune} epochs.\n")
        n_phases = self.config.n_phases or 1

        # Reset the GradScaler for AutoCast
        self.ampGradScaler = torch.cuda.amp.GradScaler(enabled=(self.config.use_amp is True))

        # Update the retrain schedule individually for every phase/cycle
        self.define_retrain_schedule(n_epochs_finetune=n_epochs_finetune,
                                     pruning_sparsity=pruning_sparsity)

        self.strategy.set_to_finetuning_phase()
        for epoch in range(1, n_epochs_finetune + 1, 1):
            self.reset_averaged_metrics()
            sys.stdout.write(
                f"\nFinetuning: phase {phase}/{n_phases} | epoch {epoch}/{n_epochs_finetune}\n")
            # Train
            t = time.time()
            self.train_epoch(data='train')
            self.evaluate_model(data='val')

            self.strategy.at_epoch_end(epoch=epoch)
            self.log(runTime=time.time() - t, finetuning=True,
                     final_logging=(epoch == n_epochs_finetune and phase == n_phases))

    def train_epoch(self, data='train', is_training=True):
        assert not (data in ['test', 'val', 'ood'] and is_training), "Can't train on test/val/ood set."
        loaderDict = {'train': self.trainLoader,
                      'val': self.valLoader,
                      'test': self.testLoader,
                      'ood': self.oodLoader}
        loader = loaderDict[data]
        if loader is None and data == 'ood':
            sys.stdout.write(f"No OOD data available. Skipping.\n")
            return

        sys.stdout.write(f"Training:\n") if is_training else sys.stdout.write(
            f"Evaluation of {data} data:\n")

        with torch.set_grad_enabled(is_training):
            with tqdm(loader, leave=True) as pbar:
                for x_input, y_target, indices in pbar:
                    # Move to CUDA if possible
                    x_input = x_input.to(self.device, non_blocking=True)
                    y_target = y_target.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad()  # Zero the gradient buffers

                    itStartTime = time.time()

                    with autocast(enabled=(self.config.use_amp is True)):
                        output = self.model.train(mode=(data == 'train'))(x_input)
                        loss = self.loss_criterion(output, y_target)

                    if is_training:
                        self.ampGradScaler.scale(loss).backward()  # Scaling + Backpropagation
                        self.ampGradScaler.step(self.optimizer)  # Optimization step
                        self.ampGradScaler.update()

                        self.strategy.after_training_iteration(it=self.trainIterationCtr,
                                                               lr=float(self.optimizer.param_groups[0]['lr']))
                        self.scheduler.step()
                        self.trainIterationCtr += 1

                    itEndTime = time.time()
                    n_img_in_iteration = len(y_target)
                    ips = n_img_in_iteration / (itEndTime - itStartTime)  # Images processed per second

                    self.metrics[data]['loss'](value=loss, weight=len(y_target))
                    self.metrics[data]['accuracy'](output, y_target)
                    self.metrics[data]['ips_throughput'](ips)
                    if data in ['val', 'test']:
                        self.metrics[data]['ece'](output, y_target)
                        self.metrics[data]['mce'](output, y_target)
                        self.metrics[data]['worst_class_accuracy'](output, y_target)

    def train(self):
        self.ampGradScaler = torch.cuda.amp.GradScaler(enabled=(self.config.use_amp is True))
        for epoch in range(self.config.n_epochs + 1):
            self.reset_averaged_metrics()
            sys.stdout.write(f"\n\nEpoch {epoch}/{self.config.n_epochs}\n")
            t = time.time()
            if epoch > 0:
                # Train
                self.train_epoch(data='train')
            self.evaluate_model(data='val')

            if epoch == self.config.n_epochs:
                # Do one complete evaluation on the test data set
                self.evaluate_model(data='test')

            self.strategy.at_epoch_end(epoch=epoch)

            self.log(runTime=time.time() - t)

        self.trained_test_accuracy = self.metrics['test']['accuracy'].compute()
        self.trained_train_loss = self.metrics['train']['loss'].compute()

    def recalibrate_bn(self):
        # Reset BN statistics
        recalibration_fraction = self.config.bn_recalibration_frac
        if self.config.bn_recalibration_frac is None or not (0 <= self.config.bn_recalibration_frac <= 1):
            recalibration_fraction = 1.
            sys.stdout.write(
                f"bn_recalibration_frac not specified or invalid ({self.config.bn_recalibration_frac}). Recalibrating BN-statistics on 100% of the training data (unshuffled).\n")

        reset_ctr = 0
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
                reset_ctr += 1
        sys.stdout.write(
            f"\nReset of {reset_ctr} BN-layers successful. Recalibrating BN-statistics on {int(recalibration_fraction * 100)}% of the training data (unshuffled).\n")
        n_batches = len(self.trainLoader_unshuffled)
        max_n_batches = int(recalibration_fraction * n_batches)
        if max_n_batches == 0: return
        it = 0
        with tqdm(self.trainLoader_unshuffled, leave=True) as pbar:
            for x_input, y_target, indices in pbar:
                # Move to CUDA if possible
                x_input = x_input.to(self.device, non_blocking=True)

                with autocast(enabled=(self.config.use_amp is True)):
                    self.model.train()(x_input)
                it += 1
                if it >= max_n_batches:
                    break

        # Free the cuda cache since it might be that the entire trainLoader is allocated
        # sys.stdout.write("Emptying cuda cache.\n")
        torch.cuda.empty_cache()
