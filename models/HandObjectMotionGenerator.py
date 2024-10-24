import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from utils.arguments import CFGS



class HandObjectMotionGenerator(nn.Module):
    def __init__(self,
            text_feat_dim,
            objs_feat_dim,
            hand_dim,
            objs_dim,
            dim=512,
        ):
        super(HandObjectMotionGenerator, self).__init__()

        self.dim = dim

        self.frame_wise_pos_encoder = PositionalEncoding(d_model=dim, encode_mode="frame-wise")
        self.agent_wise_pos_encoder = PositionalEncoding(d_model=dim, encode_mode="agent-wise")

        self.timestep_emb = TimestepEmbedding(pos_encoder=self.frame_wise_pos_encoder, hidden_dim=dim, output_dim=dim)

        self.text_feat_emb = nn.Linear(text_feat_dim, dim)
        self.objs_feat_emb = nn.Linear(objs_feat_dim, dim)

        self.lhand_emb = nn.Linear(hand_dim, dim)
        self.rhand_emb = nn.Linear(hand_dim, dim)
        self.obj_emb = nn.Linear(objs_dim, dim)
    
    def forward(self, 
        x_lhand, x_rhand, x_obj, 
        objs_feat, timesteps, text_feat):

        B = x_lhand.shape[0]

        emb = self.timestep_emb(timesteps)
        emb += self.objs_feat_emb(objs_feat)
        emb += self.text_feat_emb(text_feat)

        f_lhand = self.lhand_emb(x_lhand)
        f_rhand = self.rhand_emb(x_rhand)
        f_obj = self.obj_emb(x_obj)

        x = torch.stack((f_lhand, f_rhand, f_obj), dim=1)
        x = x.reshape(-1, B, self.dim)

        x = torch.cat((emb, x), dim=0)
        x = self.frame_wise_pos_encoder(x)
        x = self.agent_wise_pos_encoder(x)

        x_out_lhand = x[0::3]
        x_out_rhand = x[1::3]
        x_out_obj = x[2::3]

        return x_out_lhand, x_out_rhand, x_out_obj



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
            x[0] = x[0] + self.pe[0]
            x[1::3] = x[1::3] + self.pe[1:(x.shape[0] + 2)//3]
            x[2::3] = x[2::3] + self.pe[1:(x.shape[0] + 2)//3]
            x[3::3] = x[3::3] + self.pe[1:(x.shape[0] + 2)//3]
        elif self.encode_mode == "agent-wise":
            x[1::3] = x[1::3] + self.pe[1:2]
            x[2::3] = x[2::3] + self.pe[len(self.pe)//3 : len(self.pe)//3+1]
            x[3::3] = x[3::3] + self.pe[len(self.pe)*2//3 : len(self.pe)*2//3+1]
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
        return self.time_embed(self.pos_encoder.pe[timesteps]).permute(1, 0, 2)