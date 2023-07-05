# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         strategies/ensembleStrategies.py
# Description:  Strategies for building a soup.
# ===========================================================================
import sys
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import torch

from strategies import strategies as usual_strategies
from utilities.utilities import Candidate
from utilities.utilities import Utilities as Utils


#### Base Class
class EnsemblingBaseClass(usual_strategies.Dense):
    """Ensembling Base Class"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.candidate_model_list = kwargs['candidate_models']
        self.runner = kwargs['runner']
        self.selected_models = None
        self.soup_metrics = {soup_type: {} for soup_type in ['candidates', 'selected']}

    @torch.no_grad()
    def get_soup_metrics(self, soup_list: list[Candidate]):

        # Load the models
        model_list = [candidate.get_model_weights() for candidate in soup_list]

        soup_metrics = {
            'max_barycentre_distance': Utils.get_barycentre_l2_distance(model_list),
            'min_barycentre_distance': Utils.get_barycentre_l2_distance(model_list, maximize=False),
        }

        for metric_name, metric_fn in zip(['l2_distance', 'angle'], [Utils.get_l2_distance, Utils.get_angle]):
            for agg_name, agg_fn in zip(['max', 'min', 'mean'], [torch.max, torch.min, torch.mean]):
                soup_metrics[f'{agg_name}_{metric_name}'] = Utils.aggregate_group_metrics(models=model_list,
                                                                                          metric_fn=metric_fn,
                                                                                          aggregate_fn=agg_fn)
        return soup_metrics

    def collect_candidate_information(self):
        model_list = []
        metrics_dict = {split: defaultdict(list) for split in ['test', 'ood']}
        for candidate in self.candidate_model_list:
            candidate_id, candidate_file = candidate.id, candidate.file
            if self.runner.model is not None:
                del self.runner.model
                torch.cuda.empty_cache()

            state_dict = torch.load(candidate_file,
                                    map_location=torch.device('cpu'))  # Load to CPU to avoid memory overhead
            self.runner.load_soup_model(ensemble_state_dict=state_dict)
            m, _ = Utils.split_weights_and_masks(state_dict)
            model_list.append(m)
            del state_dict
            self.runner.recalibrate_bn()

            # Collect and set test/ood metrics
            for split in ['test', 'ood']:
                single_model_metrics = self.runner.evaluate_soup(data=split)
                for metric, value in single_model_metrics.items():
                    metrics_dict[split][metric].append(value)
                candidate.set_metrics(metrics=single_model_metrics, split=split)

            # Collect metrics that are needed for other strategies to perform model selection
            single_model_val_metrics = self.runner.evaluate_soup(data='val')
            candidate.set_metrics(metrics=single_model_val_metrics, split='val')

        # Collect a lot of soup metrics
        candidates_soup_metrics = self.get_soup_metrics(soup_list=self.candidate_model_list)
        self.soup_metrics['candidates'] = candidates_soup_metrics
        for split in ['test', 'ood']:
            for aggName, aggFunc in zip(['mean', 'max'], [np.mean, np.max]):
                for metric, values in metrics_dict[split].items():
                    self.soup_metrics['candidates'][f'{split}.{metric}_{aggName}'] = aggFunc(values)

        # Collect prediction ensemble metrics
        ensemble_labels = self.runner.collect_avg_output_full(data='test',
                                                              candidate_model_list=self.candidate_model_list)
        ensemble_metrics = {
            'pred_ensemble.test': self.runner.evaluate_soup(data='test', ensemble_labels=ensemble_labels)}
        self.soup_metrics['candidates'].update(ensemble_metrics)

        sys.stdout.write(f"Test accuracies of ensemble runs: {metrics_dict['test']['accuracy']}.\n")

    def create_ensemble(self, **kwargs):
        n_models = len(self.candidate_model_list)
        assert n_models >= 2, "Not enough models to ensemble"
        self.enforce_prunedness()

    @torch.no_grad()
    def enforce_prunedness(self, device=torch.device('cpu')):
        """Enforce prunedness of the model"""
        for candidate in self.candidate_model_list:
            candidate.enforce_prunedness(device=device)

    @torch.no_grad()
    def average_models(self, soup_list: list[Candidate], soup_weights: torch.Tensor = None,
                       device: torch.device = torch.device('cpu')):
        if soup_weights is None:
            soup_weights = torch.ones(len(soup_list)) / len(soup_list)
        ensemble_state_dict = OrderedDict()

        for idx, candidate in enumerate(soup_list):
            candidate_id, candidate_file = candidate.id, candidate.file
            state_dict = torch.load(candidate_file, map_location=device)
            for key, val in state_dict.items():
                factor = soup_weights[idx].item()  # No need to use tensor here
                if '_mask' in key:
                    # We dont want to average the masks, hence we skip them and add later
                    continue
                if key not in ensemble_state_dict:
                    ensemble_state_dict[
                        key] = factor * val.detach().clone()  # Important: clone otherwise we modify the tensors
                else:
                    ensemble_state_dict[
                        key] += factor * val.detach().clone()  # Important: clone otherwise we modify the tensors

        # Add the masks from the last state_dict
        for key, val in state_dict.items():
            if '_mask' in key:
                ensemble_state_dict[key] = val.detach().clone()

        return ensemble_state_dict

    def final(self):
        self.callbacks['final_log_callback']()

    def get_ensemble_metrics(self):
        if self.selected_models == 'all':
            # We have already collected the metrics for all models
            self.soup_metrics['selected'] = self.soup_metrics['candidates']
        else:
            assert self.selected_models is not None and len(self.selected_models) > 0, "No models selected for metrics."
            # Collect individual metrics for the selected models, which we already have
            metrics_dict = defaultdict(lambda
                                       : defaultdict(list))
            for split in ['test', 'ood']:
                for candidate in self.selected_models:
                    single_model_metrics = candidate.get_metrics(split=split)
                    for metric, value in single_model_metrics.items():
                        metrics_dict[split][metric].append(value)
                for aggName, aggFunc in zip(['mean', 'max'], [np.mean, np.max]):
                    for metric, values in metrics_dict[split].items():
                        self.soup_metrics['selected'][f'{split}.{metric}_{aggName}'] = aggFunc(values)

            # Collect group_metrics for the selected models
            group_metrics = self.get_soup_metrics(soup_list=self.selected_models)
            self.soup_metrics['selected'].update(group_metrics)

            # Collect prediction ensemble metrics, only for test for now
            ensemble_labels = self.runner.collect_avg_output_full(data='test',
                                                                  candidate_model_list=self.selected_models)
            ensemble_metrics = {
                'pred_ensemble.test': self.runner.evaluate_soup(data='test', ensemble_labels=ensemble_labels)}
            self.soup_metrics['selected'].update(ensemble_metrics)
        return self.soup_metrics


class UniformEnsembling(EnsemblingBaseClass):
    """Just averages all models"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @torch.no_grad()
    def create_ensemble(self, **kwargs):
        super().create_ensemble(**kwargs)

        device = torch.device('cpu')
        soup_weights = self.get_soup_weights(soup_list=self.candidate_model_list)
        ensemble_state_dict = self.average_models(soup_list=self.candidate_model_list, soup_weights=soup_weights,
                                                  device=device)
        self.selected_models = 'all'
        return ensemble_state_dict

    def get_soup_weights(self, soup_list: list[Candidate]):
        uniform_factor = 1. / len(soup_list)
        return torch.tensor([uniform_factor] * len(soup_list))


class GreedySoup(EnsemblingBaseClass):
    """Greedy approach"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @torch.no_grad()
    def create_ensemble(self, **kwargs):
        super().create_ensemble(**kwargs)
        val_accuracies = [(candidate, candidate.get_single_metric(metric='accuracy', split='val'))
                          for candidate in self.candidate_model_list]
        device = torch.device('cpu')

        # Sort the models by their validation accuracy in decreasing order
        sorted_tuples = sorted(val_accuracies, key=lambda x: x[1], reverse=True)

        ingredients_candidates = [sorted_tuples[0][0]]
        max_val_accuracy = sorted_tuples[0][1]
        for candidate, _ in sorted_tuples[1:]:
            # Check whether we benefit from adding to the soup
            ensemble_state_dict = self.average_models(soup_list=ingredients_candidates + [candidate], device=device)
            self.callbacks['load_soup_callback'](ensemble_state_dict=ensemble_state_dict)
            self.callbacks['recalibrate_bn_callback']()
            soup_metrics = self.callbacks['soup_evaluation_callback'](data='val')
            soup_val_accuracy = soup_metrics['accuracy']
            if soup_val_accuracy >= max_val_accuracy:
                ingredients_candidates = ingredients_candidates + [candidate]
                max_val_accuracy = soup_val_accuracy

        self.selected_models = ingredients_candidates
        if len(ingredients_candidates) == len(self.candidate_model_list):
            self.selected_models = 'all'
            sys.stdout.write("GreedySoup used all candidates.\n")
        else:
            sys.stdout.write(
                f"GreedySoup used candidates with ids: {[candidate.id for candidate in ingredients_candidates]}.\n")
        final_ensemble_state_dict = self.average_models(soup_list=ingredients_candidates, device=device)
        return final_ensemble_state_dict
