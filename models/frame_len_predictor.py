import torch
import torch.nn as nn

from models.components.linear import LinearLayers


class FrameLenPredictor(nn.Module):
    def __init__(self, device, text_feature_dim=512, rand_dim=64):
        super(FrameLenPredictor, self).__init__()

        self.device = device
        self.text_feature_dim = text_feature_dim
        self.rand_dim = rand_dim

        self.predictor = LinearLayers(
            in_dim=text_feature_dim+rand_dim,
            layers_out_dim=[512, 256, 128, 1], 
            activation_func='leaky_relu', 
            activation_func_param=0.02, 
            bn=False,
            sigmoid_output=True)
        
        
    def forward(self, x):
        B = x.shape[0]
        z = torch.randn((B, self.rand_dim), requires_grad=True).to(self.device)
        x = torch.cat([z, x], dim=1)

        x = self.predictor(x) * 150

        return x