# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         strategies/strategies.py
# Description:  Sparsification strategies for regular training
# ===========================================================================
import sys
from collections import OrderedDict

import torch
import torch.nn.utils.prune as prune


#### Dense Base Class
class Dense:
    """Dense base class for defining callbacks, does nothing but showing the structure and inherits."""
    required_params = []

    def __init__(self, **kwargs):
        self.masks = dict()
        self.lr_dict = OrderedDict()  # it:lr
        self.is_in_finetuning_phase = False

        self.model = kwargs['model']
        self.run_config = kwargs['config']
        self.callbacks = kwargs['callbacks']
        self.goal_sparsity = self.run_config['goal_sparsity']
        self.prune_structured = self.run_config['prune_structured']

        self.optimizer = None  # To be set
        self.n_total_iterations = None

    def after_initialization(self):
        """Called after initialization of the strategy"""
        if self.prune_structured:
            self.parameters_to_prune = [(module, 'weight') for name, module in self.model.named_modules() if
                                        hasattr(module, 'weight')
                                        and not isinstance(module.weight, type(None)) and isinstance(module,
                                                                                                     torch.nn.Conv2d)]
        else:
            self.parameters_to_prune = [(module, 'weight') for name, module in self.model.named_modules() if
                                        hasattr(module, 'weight')
                                        and not isinstance(module.weight, type(None)) and not isinstance(module,
                                                                                                         torch.nn.BatchNorm2d)]
        self.n_prunable_parameters = sum(
            getattr(module, param_type).numel() for module, param_type in self.parameters_to_prune)

    def set_optimizer(self, opt, **kwargs):
        self.optimizer = opt
        if 'n_total_iterations' in kwargs:
            self.n_total_iterations = kwargs['n_total_iterations']

    @torch.no_grad()
    def after_training_iteration(self, **kwargs):
        """Called after each training iteration"""
        if not self.is_in_finetuning_phase:
            self.lr_dict[kwargs['it']] = kwargs['lr']

    def at_train_begin(self):
        """Called before training begins"""
        pass

    def at_epoch_start(self, **kwargs):
        """Called before the epoch starts"""
        pass

    def at_epoch_end(self, **kwargs):
        """Called at epoch end"""
        pass

    def at_train_end(self, **kwargs):
        """Called at the end of training"""
        pass

    def final(self):
        # self.make_pruning_permant()
        pass

    @torch.no_grad()
    def pruning_step(self, pruning_sparsity, only_save_mask=False, compute_from_scratch=False):
        if compute_from_scratch:
            # We have to revert to weight_orig and then compute the mask
            for module, param_type in self.parameters_to_prune:
                if prune.is_pruned(module):
                    # Enforce the equivalence of weight_orig and weight
                    orig = getattr(module, param_type + "_orig").detach().clone()
                    prune.remove(module, param_type)
                    p = getattr(module, param_type)
                    p.copy_(orig)
                    del orig
        elif only_save_mask and len(self.masks) > 0:
            for module, param_type in self.parameters_to_prune:
                if (module, param_type) in self.masks:
                    prune.custom_from_mask(module, param_type, self.masks[(module, param_type)])
        if self.prune_structured:
            # We prune filters locally
            sys.stdout.write(f"\nPruning by l2 norm.")
            for module, param_type in self.parameters_to_prune:
                prune.ln_structured(module, param_type, pruning_sparsity, n=2, dim=0)
        else:
            if self.run_config['pruning_selector'] is not None and self.run_config['pruning_selector'] == 'uniform':
                # We prune each layer individually
                for module, param_type in self.parameters_to_prune:
                    prune.l1_unstructured(module, name=param_type, amount=pruning_sparsity)
            else:
                # Default: prune globally
                prune.global_unstructured(
                    self.parameters_to_prune,
                    pruning_method=self.get_pruning_method(),
                    amount=pruning_sparsity,
                )

        self.masks = dict()  # Stays empty if we use regular pruning
        if only_save_mask:
            for module, param_type in self.parameters_to_prune:
                if prune.is_pruned(module):
                    # Save the mask
                    mask = getattr(module, param_type + '_mask')
                    self.masks[(module, param_type)] = mask.detach().clone()
                    setattr(module, param_type + '_mask', torch.ones_like(mask))
                    # Remove (i.e. make permanent) the reparameterization
                    prune.remove(module=module, name=param_type)
                    # Delete the temporary mask to free memory
                    del mask

    def enforce_prunedness(self):
        """
        Makes the pruning permant, i.e. set the pruned weights to zero, than reinitialize from the same mask
        This ensures that we can actually work (i.e. LMO, rescale computation) with the parameters
        Important: For this to work we require that pruned weights stay zero in weight_orig over training
        hence training, projecting etc should not modify (pruned) 0 weights in weight_orig
        """
        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                # Save the mask
                mask = getattr(module, param_type + '_mask')
                # Remove (i.e. make permanent) the reparameterization
                prune.remove(module=module, name=param_type)
                # Reinitialize the pruning
                prune.custom_from_mask(module=module, name=param_type, mask=mask)
                # Delete the temporary mask to free memory
                del mask

    def prune_momentum(self):
        opt_state = self.optimizer.state
        for module, param_type in self.parameters_to_prune:
            if prune.is_pruned(module):
                # Enforce the prunedness of momentum buffer
                param_state = opt_state[getattr(module, param_type + "_orig")]
                if 'momentum_buffer' in param_state:
                    mask = getattr(module, param_type + "_mask")
                    param_state['momentum_buffer'] *= mask.to(dtype=param_state['momentum_buffer'].dtype)

    def get_pruning_method(self):
        raise NotImplementedError("Dense has no pruning method, this must be implemented in each child class.")

    @torch.no_grad()
    def make_pruning_permanent(self):
        """Makes the pruning permanent and removes the pruning hooks"""
        # Note: this does not remove the pruning itself, but rather makes it permanent
        if len(self.masks) == 0:
            for module, param_type in self.parameters_to_prune:
                if prune.is_pruned(module):
                    prune.remove(module, param_type)
        else:
            for module, param_type in self.masks:
                # Get the mask
                mask = self.masks[(module, param_type)]

                # Apply the mask
                orig = getattr(module, param_type)
                orig *= mask
            self.masks = dict()

    def set_to_finetuning_phase(self):
        self.is_in_finetuning_phase = True


class IMP(Dense):
    """Iterative Magnitude Pruning Base Class"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.phase = self.run_config['phase']
        self.n_phases = self.run_config['n_phases']
        self.n_epochs_per_phase = self.run_config['n_epochs_per_phase']

    def at_train_end(self, **kwargs):
        # Sparsity factor on remaining weights after each round, yields desired_sparsity after all rounds
        prune_per_phase = 1 - (1 - self.goal_sparsity) ** (1. / self.n_phases)
        phase = self.phase
        self.pruning_step(pruning_sparsity=prune_per_phase)
        self.current_sparsity = 1 - (1 - prune_per_phase) ** phase
        self.callbacks['after_pruning_callback']()
        self.finetuning_step(pruning_sparsity=prune_per_phase, phase=phase)

    def finetuning_step(self, pruning_sparsity, phase):
        self.callbacks['finetuning_callback'](pruning_sparsity=pruning_sparsity,
                                              n_epochs_finetune=self.n_epochs_per_phase,
                                              phase=phase)

    def get_pruning_method(self):
        if self.run_config['pruning_selector'] in ['global', 'uniform']:
            # For uniform this is not actually needed, we always select using L1
            return prune.L1Unstructured
        elif self.run_config['pruning_selector'] == 'random':
            return prune.RandomUnstructured
        else:
            raise NotImplementedError

    def final(self):
        super().final()
        self.callbacks['final_log_callback']()
