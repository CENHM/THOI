import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.arguments import CFGS


class TNet(nn.Module):
    def __init__(self, k=64, device=None):
        super(TNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k
        self.device = device

    def forward(self, x):
        B, D, N = x.shape

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        I = torch.eye(self.k).float().to(self.device)
        I = I.repeat(B, 1, 1)
        x = x.reshape(B, self.k, self.k)
        x = x + I
        
        return x


class PointNet(nn.Module):
    def __init__(self, init_k, local_feat=True, device=None):
        super(PointNet, self).__init__()

        self.stn1 = TNet(k=init_k, device=device)
        if CFGS.second_stn:
            self.stn2 = TNet(k=64, device=device)

        self.conv1 = torch.nn.Conv1d(init_k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.local_feat = local_feat
        

    def forward(self, x): # B, N, D
        x = x.transpose(2, 1) # B, D, N

        tx = self.stn1(x)  # B, D, D
        x = x.transpose(2, 1)
        x = torch.bmm(x, tx)
        x = x.transpose(2, 1) # B, D, N

        x = F.relu(self.bn1(self.conv1(x)))

        if CFGS.second_stn:
            tx = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, tx) # B, N, D
            x = x.transpose(2, 1)
        
        local_feature = x.transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        global_feature = torch.max(x, 2, keepdim=True)[0].reshape(-1, 1024)

        if self.local_feat:
            return local_feature, global_feature
        return global_feature
