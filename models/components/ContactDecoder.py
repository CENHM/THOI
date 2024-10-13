import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.arguments import CFGS


class ContactDecoder(nn.Module):
    def __init__(self):
        super(ContactDecoder, self).__init__()

        self.fc1 = torch.nn.Linear(1665, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 1)

    def forward(self, x): # B, N, D
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = self.fc4(x)

        return x
