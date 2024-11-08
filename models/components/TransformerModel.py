import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from utils.arguments import CFGS



class PositionalEncoding(nn.Module):
    def __init__(self, 
            d_model,
            comp,
            dropout=0.1, 
            max_frame_len=CFGS.max_length,
            encode_mode="default"
        ):
        super(PositionalEncoding, self).__init__()

        assert comp == "homg" or comp == "hrn"
        self.n_comp = 3 if comp == "homg" else 2
        self.encode_mode = encode_mode
        max_len = self.n_comp * max_frame_len

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.encode_mode == "frame-wise":
            if self.n_comp == 3:
                x[:, 0] += self.pe[0]
                x[:, 1::3] += self.pe[1]
                x[:, 2::3] += self.pe[2]
                x[:, 3::3] += self.pe[3]
            else: # self.n_comp == 2
                x[:, 0::2] += self.pe[0]
                x[:, 1::2] += self.pe[1]
        elif self.encode_mode == "agent-wise":
            start = 1 if self.n_comp == 3 else 0
            for i in range(start, x.shape[1], self.n_comp):
                x[:, i:i+self.n_comp] += self.pe[i // self.n_comp]
        elif self.encode_mode == "default":
            x += self.pe[:x.shape[1]]
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
            n_head=CFGS.n_head,
            n_layers=CFGS.n_layers,
            activation="gelu",
        ):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                        nhead=n_head, 
                                                        dim_feedforward=d_model*2,
                                                        activation=activation)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x