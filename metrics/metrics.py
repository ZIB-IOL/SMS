# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         metrics/metrics.py
# Description:  Useful metrics
# ===========================================================================
import math
from typing import Union, Tuple, List

import torch

from metrics import flops


@torch.no_grad()
def get_flops(model, x_input):
    return flops.flops(model, x_input)


@torch.no_grad()
def get_theoretical_speedup(n_flops: int, n_nonzero_flops: int) -> dict:
    if n_nonzero_flops == 0:
        # Would yield infinite speedup
        return {}
    return float(n_flops) / n_nonzero_flops


def modular_sparsity(parameters_to_prune: List) -> float:
    """Returns the global sparsity out of all prunable parameters"""
    n_total, n_zero = 0., 0.
    for module, param_type in parameters_to_prune:
        if hasattr(module, param_type) and not isinstance(getattr(module, param_type), type(None)):
            param = getattr(module, param_type)
            n_param = float(torch.numel(param))
            n_zero_param = float(torch.sum(param == 0))
            n_total += n_param
            n_zero += n_zero_param
    return float(n_zero) / n_total if n_total > 0 else 0


def global_sparsity(module: torch.nn.Module, param_type: Union[str, None] = None) -> float:
    """Returns the global sparsity of module (mostly of entire model)"""
    n_total, n_zero = 0., 0.
    param_list = ['weight', 'bias'] if not param_type else [param_type]
    for name, module in module.named_modules():
        for param_type in param_list:
            if hasattr(module, param_type) and not isinstance(getattr(module, param_type), type(None)):
                param = getattr(module, param_type)
                n_param = float(torch.numel(param))
                n_zero_param = float(torch.sum(param == 0))
                n_total += n_param
                n_zero += n_zero_param
    return float(n_zero) / n_total


@torch.no_grad()
def get_parameter_count(model: torch.nn.Module) -> Tuple[int, int]:
    n_total = 0
    n_nonzero = 0
    param_list = ['weight', 'bias']
    for name, module in model.named_modules():
        for param_type in param_list:
            if hasattr(module, param_type) and not isinstance(getattr(module, param_type), type(None)):
                p = getattr(module, param_type)
                n_total += int(p.numel())
                n_nonzero += int(torch.sum(p != 0))
    return n_total, n_nonzero


@torch.no_grad()
def get_distance_to_pruned(model: torch.nn.Module, sparsity: float) -> Tuple[float, float]:
    prune_vector = torch.cat(
        [module.weight.flatten() for name, module in model.named_modules() if hasattr(module, 'weight')
         and not isinstance(module.weight, type(None)) and not isinstance(module,
                                                                          torch.nn.BatchNorm2d)])
    n_params = float(prune_vector.numel())
    k = int((1 - sparsity) * n_params)
    total_norm = float(torch.norm(prune_vector, p=2))
    pruned_norm = float(torch.norm(torch.topk(torch.abs(prune_vector), k=k).values, p=2))
    distance_to_pruned = math.sqrt(abs(total_norm ** 2 - pruned_norm ** 2))
    rel_distance_to_pruned = distance_to_pruned / total_norm if total_norm > 0 else 0
    return distance_to_pruned, rel_distance_to_pruned


@torch.no_grad()
def get_distance_to_origin(model: torch.nn.Module) -> float:
    prune_vector = torch.cat(
        [module.weight.flatten() for name, module in model.named_modules() if hasattr(module, 'weight')
         and not isinstance(module.weight, type(None)) and not isinstance(module,
                                                                          torch.nn.BatchNorm2d)])
    return float(torch.norm(prune_vector, p=2))


def per_layer_sparsity(model: torch.nn.Module):
    """Returns the per-layer-sparsity of model"""
    per_layer_sparsity_dict = dict()
    param_type = 'weight'  # Only compute for weights, since we do not sparsify biases
    for name, submodule in model.named_modules():
        if hasattr(submodule, param_type) and not isinstance(getattr(submodule, param_type), type(None)):
            if name in per_layer_sparsity_dict:
                continue
            per_layer_sparsity_dict[name] = global_sparsity(submodule, param_type=param_type)
    return per_layer_sparsity_dict
