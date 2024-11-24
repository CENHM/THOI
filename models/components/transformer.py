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
            max_len=5000,
            encode_mode="default"
        ):
        super(PositionalEncoding, self).__init__()

        assert comp == "homg" or comp == "hrn"
        self.n_comp = 3 if comp == "homg" else 2
        self.encode_mode = encode_mode

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.encode_mode == "frame-wise":
            if self.n_comp == 3:
                x[:, 0] += self.pe[0]

                a = x[:, 1::3]
                b = self.pe[1:(x.shape[1] + 2) // 3]

                x[:, 1::3] += self.pe[1:(x.shape[1] + 2) // 3].squeeze(1)
                x[:, 2::3] += self.pe[1:(x.shape[1] + 2) // 3].squeeze(1)
                x[:, 3::3] += self.pe[1:(x.shape[1] + 2) // 3].squeeze(1)
            else: # self.n_comp == 2
                x[:, 0::2] += self.pe[0:x.shape[1] // 2].squeeze(1)
                x[:, 1::2] += self.pe[0:x.shape[1] // 2].squeeze(1)
        elif self.encode_mode == "agent-wise":
            if self.n_comp == 3:   
                x[:, 1::3] += self.pe[1:2]
                x[:, 2::3] += self.pe[len(self.pe)//3:len(self.pe)//3+1].squeeze(1)
                x[:, 3::3] += self.pe[len(self.pe)*2//3:len(self.pe)*2//3+1].squeeze(1)
            else:
                x[:, 0::2] += self.pe[0:1]
                x[:, 1::2] += self.pe[len(self.pe)//2:len(self.pe)//2+1].squeeze(1)
        elif self.encode_mode == "default":
            x += self.pe[0:x.shape[1]]
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
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model * 2,
            activation=activation,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=n_layers,
        )

    def forward(self, x, padding_mask):
        x = self.encoder(x, src_key_padding_mask=padding_mask.squeeze(-1))
        return x