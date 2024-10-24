import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.arguments import CFGS


class TransformerEncoder(nn.Module):
    def __init__(self,
        d_model=512,
        n_head=4,
        num_layers=8,
        activation="gelu",
    ):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                        nhead=n_head, 
                                                        dim_feedforward=d_model*2,
                                                        activation=activation)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x