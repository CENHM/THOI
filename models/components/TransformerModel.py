import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from utils.arguments import CFGS



class PositionalEncoding(nn.Module):
    def __init__(self, 
        d_model, 
        dropout=0.1, 
        max_len=5000,
        encode_mode="default"
    ):
        super(PositionalEncoding, self).__init__()

        self.encode_mode = encode_mode

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.encode_mode == "frame-wise":
            x[:, 0] = x[:, 0] + self.pe[0]
            x[:, 1::3] = x[:, 1::3] + self.pe[1:1+(x.shape[0] + 2)//3, 0, :]
            x[:, 2::3] = x[:, 2::3] + self.pe[1:1+(x.shape[0] + 2)//3, 0, :]
            x[:, 3::3] = x[:, 3::3] + self.pe[1:1+(x.shape[0] + 2)//3, 0, :]
        elif self.encode_mode == "agent-wise":
            x[:, 1::3] = x[:, 1::3] + self.pe[1:2, 0, :]
            x[:, 2::3] = x[:, 2::3] + self.pe[len(self.pe)//3 : len(self.pe)//3+1, 0, :]
            x[:, 3::3] = x[:, 3::3] + self.pe[len(self.pe)*2//3 : len(self.pe)*2//3+1, 0, :]
        elif self.encode_mode == "default":
            x = x + self.pe[:x.shape[0], :]
        else:
            raise ValueError(f"unknown position encoding mode of: {self.encode_mode}")
        return self.dropout(x)



class TimestepEmbedding(nn.Module):
    def __init__(self, 
        pos_encoder,
        hidden_dim, 
        output_dim,
    ):
        super().__init__()
        self.pos_encoder = pos_encoder

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.pos_encoder.pe[timesteps])
    

class TransformerEncoder(nn.Module):
    def __init__(self,
        d_model,
        n_head,
        num_layers,
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