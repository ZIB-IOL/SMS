# ===========================================================================
# Project:      Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging - IOL Lab @ ZIB
# Paper:        arxiv.org/abs/2306.16788
# File:         metrics/flops.py
# Description:  Methods to compute Inference-FLOPS. Modified from https://github.com/JJGO/shrinkbench
# ===========================================================================
from collections import OrderedDict

import numpy as np
import torch


@torch.no_grad()
def forward_hook_applyfn(hook, model):
    """Modified from https://github.com/JJGO/shrinkbench"""
    hooks = []

    def register_hook(module):
        if (
                not isinstance(module, torch.nn.Sequential)
                and
                not isinstance(module, torch.nn.ModuleList)
                and
                not isinstance(module, torch.nn.ModuleDict)
                and
                not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    return register_hook, hooks


@torch.no_grad()
def get_flops_on_activations(model, x_input):
    flops_on_activations = OrderedDict()
    FLOP_fn = {
        torch.nn.Conv2d: _conv2d_flops,
        torch.nn.Linear: _linear_flops,
    }

    def store_flops(module, input, output):
        if isinstance(module, torch.nn.ReLU):
            return
        assert module not in flops_on_activations, \
            f"{module} already in flops_on_activations"
        if module.__class__ in FLOP_fn:
            module_flops = FLOP_fn[module.__class__](module=module, activation=input[0])
            flops_on_activations[module] = int(module_flops)

    fn, hooks = forward_hook_applyfn(store_flops, model)
    model.apply(fn)
    with torch.no_grad():
        model.eval()(x_input)

    for h in hooks:
        h.remove()

    return flops_on_activations


@torch.no_grad()
def dense_flops(in_neurons, out_neurons):
    """Compute the number of multiply-adds used by a Dense (Linear) layer"""
    return in_neurons * out_neurons


@torch.no_grad()
def conv2d_flops(in_channels, out_channels, input_shape, kernel_shape,
                 padding='same', strides=1, dilation=1):
    """Compute the number of multiply-adds used by a Conv2D layer
    Args:
        in_channels (int): The number of channels in the layer's input
        out_channels (int): The number of channels in the layer's output
        input_shape (int, int): The spatial shape of the rank-3 input tensor
        kernel_shape (int, int): The spatial shape of the rank-4 kernel
        padding ({'same', 'valid'}): The padding used by the convolution
        strides (int) or (int, int): The spatial stride of the convolution;
            two numbers may be specified if it's different for the x and y axes
        dilation (int): Must be 1 for now.
    Returns:
        int: The number of multiply-adds a direct convolution would require
        (i.e., no FFT, no Winograd, etc)
    """
    # validate + sanitize input
    assert in_channels > 0
    assert out_channels > 0
    assert len(input_shape) == 2
    assert len(kernel_shape) == 2
    padding = padding.lower()
    assert padding in ('same', 'valid', 'zeros'), "Padding must be one of same|valid|zeros"
    try:
        strides = tuple(strides)
    except TypeError:
        # if one number provided, make it a 2-tuple
        strides = (strides, strides)
    assert dilation == 1 or all(d == 1 for d in dilation), "Dilation > 1 is not supported"

    # compute output spatial shape
    # based on TF computations https://stackoverflow.com/a/37674568
    if padding in ['same', 'zeros']:
        out_nrows = np.ceil(float(input_shape[0]) / strides[0])
        out_ncols = np.ceil(float(input_shape[1]) / strides[1])
    else:  # padding == 'valid'
        out_nrows = np.ceil((input_shape[0] - kernel_shape[0] + 1) / strides[0])  # noqa
        out_ncols = np.ceil((input_shape[1] - kernel_shape[1] + 1) / strides[1])  # noqa
    output_shape = (int(out_nrows), int(out_ncols))

    # work to compute one output spatial position
    nflops = in_channels * out_channels * int(np.prod(kernel_shape))

    # total work = work per output position * number of output positions
    return nflops * int(np.prod(output_shape))


@torch.no_grad()
def _conv2d_flops(module, activation):
    # Auxiliary func to use abstract flop computation

    # Drop batch & channels. Channels can be dropped since
    # unlike shape they have to match to in_channels
    input_shape = activation.shape[2:]
    return conv2d_flops(in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        input_shape=input_shape,
                        kernel_shape=module.kernel_size,
                        padding=module.padding_mode,
                        strides=module.stride,
                        dilation=module.dilation)


@torch.no_grad()
def _linear_flops(module, activation):
    # Auxiliary func to use abstract flop computation
    return dense_flops(module.in_features, module.out_features)


@torch.no_grad()
def flops(model, x_input):
    """Compute Multiply-add FLOPs estimate from model
    Arguments:
        model {torch.nn.Module} -- Module to compute flops for
        x_input {torch.Tensor} -- Input tensor needed for activations
    Returns:
        tuple:
        - int - Number of total FLOPs
        - int - Number of FLOPs related to nonzero parameters
    """

    total_flops = nonzero_flops = 0
    flops_on_activations = get_flops_on_activations(model, x_input)

    # The ones we need for backprop
    for m, module_flops in flops_on_activations.items():
        total_flops += module_flops
        # For our operations, all weights are symmetric so we can just
        # do simple rule of three for the estimation
        nonzero_flops += module_flops * float(torch.sum(m.weight != 0.0)) / float(m.weight.numel())

    return int(total_flops), int(nonzero_flops)
