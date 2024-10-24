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

