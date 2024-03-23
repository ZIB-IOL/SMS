## [ICLR24] Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging

*Authors: [Max Zimmer](https://maxzimmer.org/), [Christoph Spiegel](http://www.christophspiegel.berlin/), [Sebastian Pokutta](http://www.pokutta.com/)*

This repository contains the code to reproduce the experiments from the ICLR24 paper ["Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging"](https://arxiv.org/abs/2306.16788).
The code is based on [PyTorch 1.9](https://pytorch.org/) and the experiment-tracking platform [Weights & Biases](https://wandb.ai).

### Structure and Usage

Experiments are started from the following file:

- [`main.py`](main.py): Starts experiments using the dictionary format of Weights & Biases.

The rest of the project is structured as follows:

- [`strategies`](strategies): Contains the strategies used for training, pruning and model averaging.
- [`runners`](runners): Contains classes to control the training and collection of metrics.
- [`metrics`](metrics): Contains all metrics as well as FLOP computation methods.
- [`models`](models): Contains all model architectures used.
- [`utilities`](models): Contains useful auxiliary functions and classes.

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
