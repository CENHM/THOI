import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from models.components.TransformerModel import PositionalEncoding, TimestepEmbedding, TransformerEncoder
from models.components.DDPM import DDPM
from models.components.Linear import LinearLayers

from utils.arguments import CFGS

from utils.utils import hand_type_selection



class HandObjectMotionGenerator(nn.Module):
    def __init__(self,
            device,
            cfgs,
            inference,

            text_feat_dim=512,
            obj_feat_dim=2049,
            hand_motion_dim=99, 
            obj_motion_dim=10
        ):
        super(HandObjectMotionGenerator, self).__init__()

        self.inference = inference
        self.hand_motion_dim = hand_motion_dim
        self.obj_motion_dim = obj_motion_dim

        self.transformer_model = TransformerModel(
            device=device,
            text_feat_dim=text_feat_dim,
            obj_feat_dim=obj_feat_dim,
            hand_motion_dim=hand_motion_dim,
            obj_motion_dim=obj_motion_dim
        )
        self.diffusion_model = DDPM(
            device=device,
            denoise_model=TransformerModel,
            cfgs=cfgs,
            denoise_model=self.transformer_model
        )

    
    def forward(self, 
        lhand_motion, rhand_motion, obj_motion, obj_feat, text_feat
        ):

        if self.inference:
            pred_lhand_motion, pred_rhand_motion, pred_obj_motion = self.diffusion_model(
                lhand_motion, rhand_motion, obj_motion, 
                obj_feat, text_feat
            )
        else:
            pred_lhand_motion, pred_rhand_motion, pred_obj_motion = self.diffusion_model.sampling(
                obj_feat, text_feat, 
                self.hand_motion_dim, self.obj_motion_dim
            )
        
        return pred_lhand_motion, pred_rhand_motion, pred_obj_motion



# /lib/networks/cvae.py -> SeqCVAE.decode


class TransformerModel(nn.Module):
    def __init__(self,
            device,
            text_feat_dim,
            obj_feat_dim,
            hand_motion_dim,
            obj_motion_dim,
            dim=512,
        ):
        super(TransformerModel, self).__init__()

        self.dim = dim
        self.device = device

        self.frame_wise_pos_encoder = PositionalEncoding(d_model=dim, comp="homg", encode_mode="frame-wise")
        self.agent_wise_pos_encoder = PositionalEncoding(d_model=dim, comp="homg", encode_mode="agent-wise")

        self.timestep_emb = TimestepEmbedding(pos_encoder=self.frame_wise_pos_encoder, hidden_dim=dim, output_dim=dim)

        self.encoder = TransformerEncoder(d_model=dim, n_head=4, n_layers=8)


        self.length_prediction = LinearLayers(
            in_dim=576, # 512 + 64
            layers_out_dim=[512, 256, 128, 1], 
            activation_func='leaky_relu', 
            activation_func_param=0.02, 
            bn=False
        )


        self.text_feat_emb = nn.Linear(text_feat_dim, dim)
        self.objs_feat_emb = nn.Linear(obj_feat_dim, dim)

        self.lhand_emb = nn.Linear(hand_motion_dim, dim)
        self.rhand_emb = nn.Linear(hand_motion_dim, dim)
        self.obj_emb = nn.Linear(obj_motion_dim, dim)

        self.lhand_out_emb = nn.Linear(dim, hand_motion_dim)
        self.rhand_out_emb = nn.Linear(dim, hand_motion_dim)
        self.obj_out_emb = nn.Linear(dim, obj_motion_dim)
    
    def forward(self, 
        x_lhand, x_rhand, x_obj, 
        objs_feat, timesteps, text_feat):

        B = x_lhand.shape[0]


        z = torch.randn((B, 64), device=self.device)
        seq_length = self.length_prediction(torch.cat([z, text_feat], dim=1))


        objs_feat = objs_feat.unsqueeze(dim=1)
        text_feat = text_feat.unsqueeze(dim=1)

        emb = self.timestep_emb(timesteps)
        emb += self.objs_feat_emb(objs_feat)
        emb += self.text_feat_emb(text_feat)

        f_lhand = self.lhand_emb(x_lhand)
        f_rhand = self.rhand_emb(x_rhand)
        f_obj = self.obj_emb(x_obj)

        x = torch.stack((f_lhand, f_rhand, f_obj), dim=1)
        x = x.reshape(B, -1, self.dim)

        x = torch.cat((emb, x), dim=1)
        x = self.frame_wise_pos_encoder(x)
        x = self.agent_wise_pos_encoder(x)

        h_star_idx = hand_type_selection(text_feat, device=self.device)

        for i in range(x.shape[0]):
            if h_star_idx[i] == 0:
                x[:, 1::3] *= 0
            elif h_star_idx[i] == 1:
                x[:, 2::3] *= 0

        x = self.encoder(x)
        
        for i in range(x.shape[0]):
            if h_star_idx[i] == 0:
                x[:, 1::3] *= 0
            elif h_star_idx[i] == 1:
                x[:, 2::3] *= 0


        x_out_lhand = self.lhand_out_emb(x[:, 1::3])
        x_out_rhand = self.rhand_out_emb(x[:, 2::3])
        x_out_obj = self.obj_out_emb(x[:, 3::3])

        return x_out_lhand, x_out_rhand, x_out_obj
