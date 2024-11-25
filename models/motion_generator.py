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
            device,
            text_feat_dim,
            obj_feat_dim,
            hand_motion_dim,
            obj_motion_dim
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
        hand_motion_mask, frame_padding_mask,
        motion_lhand=None, motion_rhand=None, motion_obj=None
        ):

        if not inference:
            return self.diffusion_model(
                motion_lhand, motion_rhand, motion_obj, 
                obj_feat, text_feat,
                hand_motion_mask, frame_padding_mask,
                frame_len=frame_len
            )
        else:
            return self.diffusion_model.sampling(
                obj_feat, text_feat, 
                hand_motion_mask, frame_padding_mask,
                self.hand_motion_dim, self.obj_motion_dim,
                frame_len
            )


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
        hand_motion_mask, frame_padding_mask,
        timesteps,
        ):

        mask_lhand = hand_motion_mask["mask_lhand"]
        mask_rhand = hand_motion_mask["mask_rhand"]

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
        x[:, 1::3] = x[:, 1::3] * mask_lhand * frame_padding_mask
        x[:, 2::3] = x[:, 2::3] * mask_rhand * frame_padding_mask
        x[:, 3::3] = x[:, 3::3] * frame_padding_mask

        key_mask = frame_padding_mask.repeat_interleave(3, dim=1).to(self.device)
        key_mask = torch.cat([
            torch.full((B, 1, 1), 1.).to(self.device),
            key_mask], dim=1)
        key_mask = torch.where(key_mask!=0., False, True)

        x = self.encoder(x, key_mask)
        
        x_out_lhand = self.lhand_out_emb(x[:, 1::3])
        x_out_rhand = self.rhand_out_emb(x[:, 2::3])
        x_out_obj = self.obj_out_emb(x[:, 3::3])
        
        # mask input
        x_out_lhand = x_out_lhand * mask_lhand * frame_padding_mask
        x_out_rhand = x_out_rhand * mask_rhand * frame_padding_mask
        x_out_obj = x_out_obj * frame_padding_mask

        return x_out_lhand, x_out_rhand, x_out_obj
