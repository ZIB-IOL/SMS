# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         lr_schedulers.py
# Description:  All kinds of learning rate schedulers
# ===========================================================================

import warnings
from bisect import bisect_right

import torch


class FixedLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Just uses the learning rate given by a list
    """

    def __init__(self, optimizer, lrList, last_epoch=-1):
        self.lrList = lrList

        super(FixedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.lrList[self.last_epoch] for _ in self.optimizer.param_groups]


class SequentialSchedulers(torch.optim.lr_scheduler.SequentialLR):
    """
    Repairs SequentialLR to properly use the last learning rate of the previous scheduler when reaching milestones
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(SequentialSchedulers, self).__init__(**kwargs)

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()


class ChainedSchedulers(torch.optim.lr_scheduler.ChainedScheduler):
    """
    Repairs ChainedScheduler to avoid a known bug that makes it into the pytorch release soon
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(ChainedSchedulers, self).__init__(**kwargs)
