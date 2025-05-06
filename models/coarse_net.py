# coarse_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseNet(nn.Module):
    def __init__(self):
        super(CoarseNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected regression for 3-DoF (theta, tx, ty)
        self.fc = nn.Linear(256, 3)

    def forward(self, flow):
        x = self.encoder(flow)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        params = self.fc(x)
        return params
