import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LinearLayers(nn.Module):
    def __init__(
        self, 
        in_dim, 
        layers_out_dim: list,
        activation_func=None,
        activation_func_param=None,
        bn=False
    ):
        super().__init__()
        self.MLP = nn.Sequential()

        activation_funcs = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(negative_slope=activation_func_param or 1e-2),
        }

        assert activation_func is None or activation_func in activation_funcs

        for _, (in_dim, out_dim) in enumerate(zip([in_dim] + layers_out_dim[:-2], layers_out_dim[:-1])):
            self.MLP.append(nn.Linear(in_dim, out_dim))
            if bn:
                self.MLP.append(nn.BatchNorm1d(out_dim))
            if activation_func is not None:
                self.MLP.append(activation_funcs[activation_func])
        self.MLP.append(nn.Linear(layers_out_dim[-2], layers_out_dim[-1]))

    
    def forward(self, x):
        x = self.MLP(x)
        return x