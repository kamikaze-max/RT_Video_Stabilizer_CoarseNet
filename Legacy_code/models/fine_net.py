# fine_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineNet(nn.Module):
    def __init__(self):
        super(FineNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),  # Input: optical flow
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)  # Output: smoothed flow (2 channels)
        )

    def forward(self, flow):
        x = self.encoder(flow)
        out = self.decoder(x)
        return out

