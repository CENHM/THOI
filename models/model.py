import torch
import torch.nn as nn
from torch.nn import functional as F


class THOI(nn.Module):
    def __init__(self,
            device,
            hand_dim,
            hidden_dim
        ):
        super(THOI, self).__init__()

        self.device = device
        self.dim = hidden_dim

        self.frame_wise_pos_encoder = PositionalEncoding(d_model=hidden_dim, comp="hrn", encode_mode="frame-wise")
        self.agent_wise_pos_encoder = PositionalEncoding(d_model=hidden_dim, comp="hrn", encode_mode="agent-wise")

        self.encoder = TransformerEncoder(d_model=hidden_dim)

        self.fc_out_lhand = nn.Linear(in_features=hidden_dim, out_features=hand_dim)
        self.fc_out_rhand = nn.Linear(in_features=hidden_dim, out_features=hand_dim)

    
    def forward(self, 
            x_lhand, x_rhand,
            hand_motion_mask,
            frame_padding_mask,
        ):