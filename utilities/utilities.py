# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         utilities.py
# Description:  Contains a variety of useful functions.
# ===========================================================================
import itertools
import json
import math
import os
import sys
from collections import defaultdict, OrderedDict
from typing import NamedTuple, Union

import torch
import torchmetrics
import wandb
from torchmetrics.classification import MulticlassAccuracy as Accuracy


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


class Utilities:
    """Class of utility functions"""

    @staticmethod
    @torch.no_grad()
    def get_model_norm_square(model):
        """Get L2 norm squared of parameter vector. This works for a pruned model as well."""
        squared_norm = 0.
        param_list = ['weight', 'bias']
        for name, module in model.named_modules():
            for param_type in param_list:
                if hasattr(module, param_type) and not isinstance(getattr(module, param_type), type(None)):
                    param = getattr(module, param_type)
                    squared_norm += torch.norm(param, p=2) ** 2
        return float(squared_norm)

    @staticmethod
    @torch.no_grad()
    def aggregate_group_metrics(models: list[Union[OrderedDict, torch.nn.Module]], metric_fn: callable,
                                aggregate_fn: callable) -> float:
        if len(models) == 1:
            sys.stdout.write('Warning: aggregate_group_metrics called with only one model. Returning 0.\n')
            return 0.
        for idx in range(len(models)):
            if isinstance(models[idx], torch.nn.Module):
                models[idx] = models[idx].state_dict()

        collected_vals = []
        for idx_i, idx_j in itertools.combinations(range(len(models)), 2):
            model_i, model_j = models[idx_i], models[idx_j]
            dist = metric_fn(model_i, model_j)
            collected_vals.append(dist)
        return aggregate_fn(torch.tensor(collected_vals))

    @staticmethod
    @torch.no_grad()
    def get_angle(model_a: Union[OrderedDict, torch.nn.Module], model_b: Union[OrderedDict, torch.nn.Module]) -> float:
        """Get the angle between two models given as state_dict or nn.Module"""
        model_a_dict, model_b_dict = model_a, model_b
        if isinstance(model_a, torch.nn.Module):
            model_a_dict = model_a.state_dict()
        if isinstance(model_b, torch.nn.Module):
            model_b_dict = model_b.state_dict()

        dot_product = 0.
        squared_norm_a, squared_norm_b = 0., 0.
        for pName in model_a_dict.keys():
            p_a, p_b = model_a_dict[pName].flatten(), model_b_dict[pName].flatten()
            dot_product += torch.dot(p_a, p_b).item()
            squared_norm_a += torch.dot(p_a, p_a).item()
            squared_norm_b += torch.dot(p_b, p_b).item()

        # Compute the cosine similarity
        cos_sim = dot_product / (math.sqrt(squared_norm_a) * math.sqrt(squared_norm_b))

        # Calculate the angle in degrees, but first clamp the cosine similarity to [-1, 1] to avoid numerical errors
        angle_deg = math.degrees(math.acos(min(max(cos_sim, -1), 1)))
        return angle_deg

    @staticmethod
    @torch.no_grad()
    def get_l2_distance(model_a: Union[OrderedDict, torch.nn.Module],
                        model_b: Union[OrderedDict, torch.nn.Module]) -> float:
        model_a_dict, model_b_dict = model_a, model_b
        if isinstance(model_a, torch.nn.Module):
            model_a_dict = model_a.state_dict()
        if isinstance(model_b, torch.nn.Module):
            model_b_dict = model_b.state_dict()

        squared_norm = 0
        for pName in model_a_dict.keys():
            p_a, p_b = model_a_dict[pName], model_b_dict[pName]
            squared_norm += torch.norm((p_a - p_b).float(), p=2) ** 2
        return float(torch.sqrt(squared_norm))

    @staticmethod
    @torch.no_grad()
    def get_barycentre_l2_distance(models: list[Union[OrderedDict, torch.nn.Module]], maximize=True):
        """Get the distance between the barycentre of the models and the model with the largest distance to the barycentre.
        :param models: list of models given as state_dict or nn.Module
        :param maximize: if True, return the maximum distance, else return the minimum distance
        :return: the distance between the barycentre of the models and the model with the largest distance to the barycentre.
        """
        if len(models) == 1: return 0.
        for idx in range(len(models)):
            if isinstance(models[idx], torch.nn.Module):
                models[idx] = models[idx].state_dict()

        # Compute the barycentre of all models
        factor = 1. / len(models)
        barycentre = OrderedDict()

        for model_state_dict in models:
            for key, val in model_state_dict.items():
                if key not in barycentre:
                    barycentre[key] = val.detach().clone()  # Important: clone otherwise we modify the tensors
                else:
                    barycentre[key] += val.detach().clone()  # Important: clone otherwise we modify the tensors

        for key, val in barycentre.items():
            barycentre[key] = barycentre[key] * factor

        distances = []
        for idx_i, model_a in enumerate(models):
            dist = Utilities.get_l2_distance(model_a, barycentre)
            distances.append(dist)

        if maximize:
            return max(distances)
        else:
            return min(distances)

    @staticmethod
    def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
        perm_to_axes = defaultdict(list)
        for wk, axis_perms in axes_to_perm.items():
            for axis, perm in enumerate(axis_perms):
                if perm is not None:
                    perm_to_axes[perm].append((wk, axis))
        return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)

    @staticmethod
    def dump_dict_to_json_wandb(dumpDict, name):
        """Dump some dict to json and upload it"""
        fPath = os.path.join(wandb.run.dir, f'{name}.json')
        with open(fPath, 'w') as fp:
            json.dump(dumpDict, fp)
        wandb.save(fPath)

    @staticmethod
    def get_overloaded_dataset(OriginalDataset):
        class AlteredDatasetWrapper(OriginalDataset):

            def __init__(self, *args, **kwargs):
                super(AlteredDatasetWrapper, self).__init__(*args, **kwargs)

            def __getitem__(self, index):
                # Overload this to collect the class indices once in a vector, which can then be used in the sampler
                image, label = super(AlteredDatasetWrapper, self).__getitem__(index=index)
                return image, label, index

        AlteredDatasetWrapper.__name__ = OriginalDataset.__name__
        return AlteredDatasetWrapper

    @staticmethod
    def split_weights_and_masks(model):
        weights, masks = OrderedDict(), OrderedDict()

        for key, value in model.items():
            if '_mask' in key:
                name = key.replace('_mask', '')
                masks[name] = value
            elif '_orig' in key:
                name = key.replace('_orig', '')
                weights[name] = value
            else:
                weights[key] = value
        return weights, masks

    @staticmethod
    def join_weights_and_masks(weights, masks):
        state_dict = OrderedDict()
        for key, value in weights.items():
            state_dict[key + '_orig'] = value
        for key, value in masks.items():
            state_dict[key + '_mask'] = value
        return state_dict


class WorstClassAccuracy(Accuracy):
    def __init__(self, **kwargs):
        super().__init__(average=None, **kwargs)

    def compute(self):
        class_accuracies = super().compute()
        return class_accuracies.min()


class CalibrationError(torchmetrics.Metric):
    def __init__(self, num_bins=15, norm='l1', dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_bins = num_bins
        self.norm = norm
        self.add_state("bin_boundaries", default=torch.linspace(0, 1, num_bins + 1), dist_reduce_fx=None)
        self.add_state("bin_conf_sums", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("bin_correct_sums", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("bin_total_count", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Transform the predictions into probabilities
        preds = torch.softmax(preds, dim=1)

        # Compute the maximum probability for each prediction
        max_probs, max_classes = preds.max(dim=1)

        # Check if the predicted class matches the target
        correct = (max_classes == targets).float()

        # Compute the confidence for each prediction
        confidences = max_probs

        # Map confidences to the corresponding bins
        bin_indices = torch.bucketize(confidences, self.bin_boundaries[:-1]) - 1

        # Ensure that the bin indices are in the correct range
        bin_indices = bin_indices.clamp(min=0, max=self.num_bins - 1)

        # Update the bin sums and counts
        for bin_idx in range(self.num_bins):
            mask = bin_indices == bin_idx
            self.bin_conf_sums[bin_idx] += (mask * confidences).sum()
            self.bin_correct_sums[bin_idx] += (mask * correct).sum()
            self.bin_total_count[bin_idx] += mask.sum()

        # Update the total count
        self.total_count += preds.shape[0]

    def compute(self):
        assert self.total_count.item() == self.bin_total_count.sum()
        # Compute the bin accuracies and confidences
        bin_accuracies = self.bin_correct_sums / self.bin_total_count.clamp(min=1)
        bin_confidences = self.bin_conf_sums / self.bin_total_count.clamp(min=1)

        abs_errors = torch.abs(bin_accuracies - bin_confidences)
        rel_freq = self.bin_total_count / self.total_count
        if self.norm == 'l1':
            ece = torch.sum(abs_errors * rel_freq)
        elif self.norm == 'max':
            ece = torch.max(abs_errors)
        else:
            raise ValueError("Invalid norm. Supported norms are 'l1' and 'max'.")
        return ece


class Candidate(object):
    """Candidate for ensembling."""

    def __init__(self, candidate_id, candidate_file, candidate_run):
        self.id = candidate_id
        self.file = candidate_file
        self.run = candidate_run

        self._candidate_metrics = defaultdict(defaultdict)  # 'test'/'val'/'ood' -> {metric -> value}

    def set_metrics(self, metrics, split):
        self._candidate_metrics[split] = metrics

    def get_metrics(self, split):
        return self._candidate_metrics[split]

    def get_single_metric(self, metric, split):
        return self._candidate_metrics[split][metric]

    def get_model_weights(self):
        m = torch.load(self.file, map_location=torch.device('cpu'))
        weights, _ = Utilities.split_weights_and_masks(m)
        return weights

    def enforce_prunedness(self, device):
        state_dict = torch.load(self.file, map_location=device)
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            v_new = val  # Remains unchanged if not in _orig format
            if key.endswith("_orig"):
                # We loaded the _orig tensor and corresponding mask
                name = key.replace("_orig", "")  # Truncate the "_orig"
                if f"{name}_mask" in state_dict.keys():
                    v_new = v_new * state_dict[f"{name}_mask"]
            new_state_dict[key] = v_new

        # Save the new state dict
        torch.save(new_state_dict, self.file)
