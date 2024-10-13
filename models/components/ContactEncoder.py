import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.PointNet import PointNet

from utils.arguments import CFGS


class ContactEncoder(nn.Module):
    def __init__(self, device=None):
        super(ContactEncoder, self).__init__()

        self.pointnet_structure = PointNet(init_k=4, local_feat=False, device=device)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
    def __reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x): # B, N, D
        x = self.pointnet_structure(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        mean, var = torch.chunk(x, 2, dim=1)
        x = self.__reparameterize(mean, var)

        return x
