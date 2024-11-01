import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from models.components.TransformerModel import PositionalEncoding, TimestepEmbedding, TransformerEncoder

from utils.arguments import CFGS



class HandObjectMotionGenerator(nn.Module):
    def __init__(self,
            # text_feat_dim=512,
            # objs_feat_dim=2049,
            hand_joint=21,
            hand_dim=99,
            hand_joint_dim=3,
            cat_dim=2273,
            n_point=CFGS.fps_npoint,
            hidden_dim=512,
        ):
        super(HandObjectMotionGenerator, self).__init__()

        self.dim = hidden_dim

        self.transformer_model = TransformerModel()
        
        self.fc_lhand = nn.Linear(in_features=cat_dim, out_features=hidden_dim)
        self.fc_rhand = nn.Linear(in_features=cat_dim, out_features=hidden_dim)


    def forward(self, 
        hand, hand_joint, m_contact, point_cloud, att_map
        ):

        B, L = hand.shape[0], int(hand.shape[1] / 2)
        
        x_lhand = torch.chunk(hand, chunks=2, dim=1)[0]
        x_rhand = torch.chunk(hand, chunks=2, dim=1)[1]

        j_lhand = torch.chunk(hand_joint, chunks=2, dim=1)[0].view(B, L, -1)
        j_rhand = torch.chunk(hand_joint, chunks=2, dim=1)[1].view(B, L, -1)

        m_contact = m_contact.view(B, 1, -1).repeat(1, L, 1)
        # "computation of the norm is applied across the last dimension"
        point_cloud = point_cloud.norm(dim=-1)

        att_map_lhand, att_map_rhand = torch.chunk(att_map, chunks=2, dim=1)
        att_map_lhand = att_map_lhand.view(B, L, -1)
        att_map_rhand = att_map_rhand.view(B, L, -1) 

        x_lhand = torch.cat([x_lhand, j_lhand, m_contact, point_cloud, att_map_lhand], dim=2)
        x_rhand = torch.cat([x_rhand, j_rhand, m_contact, point_cloud, att_map_rhand], dim=2)

        # hand input embedding layers
        x_lhand = self.fc_lhand(x_lhand)
        x_rhand = self.fc_rhand(x_rhand)

        x = torch.stack((x_lhand, x_rhand), dim=2)  # shape (2, 2, 2, 2)
        x = x.view(B, 2*L, -1)


        
        return (lhand_pred_noise, rhand_pred_noise, obj_pred_noise), \
               (lhand_noise, rhand_noise, obj_noise)
    

class TransformerModel(nn.Module):
    def __init__(self,
            dim=512,
        ):
        super(TransformerModel, self).__init__()

        self.dim = dim

        self.frame_wise_pos_encoder = PositionalEncoding(d_model=dim, encode_mode="frame-wise")
        self.agent_wise_pos_encoder = PositionalEncoding(d_model=dim, encode_mode="agent-wise")

        self.timestep_emb = TimestepEmbedding(pos_encoder=self.frame_wise_pos_encoder, hidden_dim=dim, output_dim=dim)

        self.encoder = TransformerEncoder(d_model=dim, n_head=4, num_layers=8)

    
    def forward(self, 
        x_lhand, x_rhand, x_obj, 
        objs_feat, timesteps, text_feat):

        x = self.frame_wise_pos_encoder(x)
        x = self.agent_wise_pos_encoder(x)

        x = self.encoder(x)

        x_lhand = 

        x_out_lhand = self.lhand_out_emb(x[:, 1::3])
        x_out_rhand = self.rhand_out_emb(x[:, 2::3])
        x_out_obj = self.obj_out_emb(x[:, 3::3])

        return x_out_lhand, x_out_rhand, x_out_obj
