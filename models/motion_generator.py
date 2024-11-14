import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from models.components.transformer import PositionalEncoding, TimestepEmbedding, TransformerEncoder
from models.components.ddpm import DDPM


class MotionGenerator(nn.Module):
    def __init__(self,
            cfgs,
            device,
            hand_motion_mask,
            frame_padding_mask,
            text_feat_dim=512,
            obj_feat_dim=2049,
            hand_motion_dim=99, 
            obj_motion_dim=10
        ):
        super(MotionGenerator, self).__init__()

        self.device = device
        self.hand_motion_dim = hand_motion_dim
        self.obj_motion_dim = obj_motion_dim
        self.max_frame = cfgs.max_frame

        self.transformer_model = TransformerModel(
            device=device,
            hand_motion_mask=hand_motion_mask,
            frame_padding_mask=frame_padding_mask,
            text_feat_dim=text_feat_dim,
            obj_feat_dim=obj_feat_dim,
            hand_motion_dim=hand_motion_dim,
            obj_motion_dim=obj_motion_dim
        )
        self.diffusion_model = DDPM(
            device=device,
            cfgs=cfgs,
            denoise_model=self.transformer_model
        )
        

    
    def forward(self, 
        obj_feat, text_feat, 
        frame_len,
        inference,
        motion_lhand=None, motion_rhand=None, motion_obj=None
        ):

        B = motion_lhand.shape[0]
        

        if not inference:
            return self.diffusion_model(
                motion_lhand, motion_rhand, motion_obj, 
                obj_feat, text_feat,
                frame_len=frame_len
            )
        else:
            return self.diffusion_model.sampling(
                obj_feat, text_feat, 
                self.hand_motion_dim, self.obj_motion_dim,
                frame_len
            )


class TransformerModel(nn.Module):
    def __init__(self,
            device,
            hand_motion_mask,
            frame_padding_mask,
            text_feat_dim,
            obj_feat_dim,
            hand_motion_dim,
            obj_motion_dim,
            dim=512,
        ):
        super(TransformerModel, self).__init__()

        self.dim = dim
        self.device = device

        self.mask_lhand = hand_motion_mask["mask_lhand"]
        self.mask_rhand = hand_motion_mask["mask_rhand"]

        self.frame_padding_mask = torch.where(~frame_padding_mask, 1.0, 0.0)

        self.frame_wise_pos_encoder = PositionalEncoding(
            d_model=dim, comp="homg", encode_mode="frame-wise")
        self.agent_wise_pos_encoder = PositionalEncoding(
            d_model=dim, comp="homg", encode_mode="agent-wise")

        self.timestep_emb = TimestepEmbedding(
            pos_encoder=self.frame_wise_pos_encoder, 
            hidden_dim=dim, 
            output_dim=dim
        )

        self.encoder = TransformerEncoder(d_model=dim, n_head=4, n_layers=8)


        self.text_feat_emb = nn.Linear(text_feat_dim, dim)
        self.objs_feat_emb = nn.Linear(obj_feat_dim, dim)

        self.lhand_emb = nn.Linear(hand_motion_dim, dim)
        self.rhand_emb = nn.Linear(hand_motion_dim, dim)
        self.obj_emb = nn.Linear(obj_motion_dim, dim)

        self.lhand_out_emb = nn.Linear(dim, hand_motion_dim)
        self.rhand_out_emb = nn.Linear(dim, hand_motion_dim)
        self.obj_out_emb = nn.Linear(dim, obj_motion_dim)
    
    def forward(self, 
        lhand_motion, rhand_motion, obj_motion, 
        obj_feat, text_feat, 
        timesteps,
        ):

        B, L = lhand_motion.shape[:2]

        obj_feat = obj_feat.unsqueeze(dim=1)
        text_feat = text_feat.unsqueeze(dim=1)

        emb = self.timestep_emb(timesteps)
        emb += self.objs_feat_emb(obj_feat)
        emb += self.text_feat_emb(text_feat)

        f_lhand = self.lhand_emb(lhand_motion)
        f_rhand = self.rhand_emb(rhand_motion)
        f_obj = self.obj_emb(obj_motion)


        x = torch.zeros((B, 3*L, self.dim, )).to(self.device)
        x[:, 0::3] = f_lhand
        x[:, 1::3] = f_rhand
        x[:, 2::3] = f_obj
        x = torch.cat((emb, x), dim=1)

        x = self.frame_wise_pos_encoder(x)
        x = self.agent_wise_pos_encoder(x)

        # mask input
        x[:, 1::3] *= self.mask_lhand
        x[:, 2::3] *= self.mask_rhand
        x *= self.frame_padding_mask

        # not batch-first transform        
        x = x.reshape(-1, B, self.dim)
        x = self.encoder(x, self.frame_padding_mask)
        x = x.reshape(B, -1, self.dim)
        
        x_out_lhand = self.lhand_out_emb(x[:, 1::3])
        x_out_rhand = self.rhand_out_emb(x[:, 2::3])
        x_out_obj = self.obj_out_emb(x[:, 3::3])
        
        # mask input
        x_out_lhand = x_out_lhand * self.mask_lhand * self.frame_padding_mask[:, 1::3]
        x_out_rhand = x_out_rhand * self.mask_rhand * self.frame_padding_mask[:, 2::3]
        x_out_obj *= self.frame_padding_mask[:, 3::3]

        return x_out_lhand, x_out_rhand, x_out_obj
