import torch
from torch import nn

class CBLModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )       
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels * 5, out_channels, kernel_size=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        x1 = self.conv1(features)
        x2 = self.conv2(features)
        x3 = self.conv3(features)
        x4 = self.conv4(features)
        x5 = self.conv5(features)
        output = torch.concat([x1, x2, x3, x4, x5], 1)
        return self.conv6(output)
        