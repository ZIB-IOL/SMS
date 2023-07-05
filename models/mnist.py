# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         models/mnist.py
# Description:  MNIST Models
# ===========================================================================
import torch

from utilities.utilities import Utilities as Utils


class Simple(torch.nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.fc1 = torch.nn.Linear(784, 512, bias=True)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(512, 10, bias=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def get_permutation_spec():
        dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in),
                                                      f"{name}.bias": (p_out,)} if bias else {
            f"{name}.weight": (p_out, p_in)}

        return Utils.permutation_spec_from_axes_to_perm({
            **dense("fc1", None, "P_bg0", True),
            **dense("fc2", "P_bg0", None, True),
        })
