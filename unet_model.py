import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet169


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1, bias=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = DoubleConv(mid_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is chw
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.densenet = densenet169(pretrained=True)
        self.encoder = nn.Sequential(*list(self.densenet.features.children()))
        self.encoder.eval()
        self.up0 = Up(1664, 1920,832)
        self.up1 = Up(832, 960, 416)
        self.up2 = Up(416, 480, 208)
        self.up3 = Up(208, 272, 104)
        self.out = OutConv(104, 1)

    def forward(self, x):
        features = []
        with torch.no_grad():
            for i, module in enumerate(self.encoder):
                x = module(x)
                if i in [0, 3, 5, 7]:
                    features.append(x)
        x = self.up0(x, features[3])
        x = self.up1(x, features[2])
        x = self.up2(x, features[1])
        x = self.up3(x, features[0])
        x = self.out(x)
        x = x - torch.min(x.view(x.size(0), -1))
        x = x / torch.max(x.view(x.size(0), -1))
        return x
