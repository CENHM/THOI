import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from models.components.TransformerModel import TransformerModel
from models.components.Diffusion import DiffusionModel

from utils.arguments import CFGS



class HandObjectMotionGenerator(nn.Module):
    def __init__(self,
            text_feat_dim=512,
            objs_feat_dim=2049,
            hand_dim=99,
            objs_dim=10,
            dim=512,
        ):
        super(HandObjectMotionGenerator, self).__init__()

        self.transformer_model = TransformerModel(text_feat_dim,
                                                  objs_feat_dim,
                                                  hand_dim,
                                                  objs_dim,)
        self.diffusion_model = DiffusionModel(denoise_model=self.transformer_model)
    
    def forward(self, 
        x_lhand, x_rhand, x_obj, 
        timesteps, objs_feat, text_feat, training):

        (lhand_pred_noise, rhand_pred_noise, obj_pred_noise), \
            (lhand_noise, rhand_noise, obj_noise) = self.diffusion_model(x_lhand, x_rhand, x_obj, 
                                                       objs_feat, timesteps, text_feat, training)
        
        return (lhand_pred_noise, rhand_pred_noise, obj_pred_noise), \
               (lhand_noise, rhand_noise, obj_noise)

