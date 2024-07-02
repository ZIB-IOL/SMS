## [ICLR24] Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging

*Authors: [Max Zimmer](https://maxzimmer.org/), [Christoph Spiegel](http://www.christophspiegel.berlin/), [Sebastian Pokutta](http://www.pokutta.com/)*

This repository contains the code to reproduce the experiments from the ICLR24 paper ["Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging"](https://arxiv.org/abs/2306.16788).
The code is based on [PyTorch 1.9](https://pytorch.org/) and the experiment-tracking platform [Weights & Biases](https://wandb.ai). See the [blog post](https://www.pokutta.com/blog/research/2023/08/05/abstract-SMS.html) or the [twitter thread](https://x.com/maxzimmerberlin/status/1787052536442077479) for a TL;DR.

### Structure and Usage
#### Structure
Experiments are started from the following file:

- [`main.py`](main.py): Starts experiments using the dictionary format of Weights & Biases.

The rest of the project is structured as follows:

- [`strategies`](strategies): Contains the strategies used for training, pruning and model averaging.
- [`runners`](runners): Contains classes to control the training and collection of metrics.
- [`metrics`](metrics): Contains all metrics as well as FLOP computation methods.
- [`models`](models): Contains all model architectures used.
- [`utilities`](models): Contains useful auxiliary functions and classes.

#### Usage
An entire experiment is subdivided into multiple steps, each being multiple (potentially many) different runs and wandb experiments. First of all, a model has to be pretrained using the `Dense` strategy. This steps is completely agnostic to any pruning specifications. Then, for each phase or prune-retrain-cycle (specified by the `n_phases` parameter and controlled by `phase` parameter), the following steps are executed:
1. Strategy `IMP`: Prune the model using the IMP strategy. Here, it is important to specify the `ensemble_by`, `split_val` and `n_splits_total` parameters:
   - `ensemble_by`: The parameter which is varied when retraining multiple models. E.g. setting this to `weight_decay` will train multiple models with different weight decay values.
   - `split_val`: The value by which the `ensemble_by` parameter is split. E.g. setting this to 0.0001 while using `weight_decay` as `ensemble_by` will retrain a model with weight decay 0.0001, all else being equal.
   - `n_splits_total`: The total number of splits for the `ensemble_by` parameter. If set to three, the souping operation in the next step will expect three models to be present, given the `ensemble_by` configuration.
2. Strategy `Ensemble`: Souping the models. This step will average the weights of the models specified by the `ensemble_by` parameter. The `ensemble_by` parameter has to be the same as in the previous step. `n_splits_total` has to be the same as well. `split_val` is not used in this step and has to be set to None. The `ensemble_method` parameter controls how the models are averaged.

### Citation

In case you find the paper or the implementation useful for your own research, please consider citing:

```
@inproceedings{zimmer2024sparse,
title={Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging},
author={Max Zimmer and Christoph Spiegel and Sebastian Pokutta},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=xx0ITyHp3u}
}
```
