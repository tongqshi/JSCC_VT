import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels[0])

        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.convblock = ConvBlock(in_channels, out_channels)

    def forward(self, inputs):
        out = self.maxpool(inputs)
        out = self.convblock(out)
        return out

class ExpansiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels[0], kernel_size=2, stride=2, padding=0)
        self.convblock = ConvBlock(out_channels[0]*2, out_channels[1:])

    def forward(self, inputs, concat_inputs):
        x = self.upconv(inputs)

        # Copy and crop
        _, _, h, w = concat_inputs.size()
        _, _, target_h, target_w = x.size()

        h_crop = (h - target_h) // 2
        w_crop = (w - target_w) // 2

        concat_x = concat_inputs[:, :, h_crop:h_crop+target_h, w_crop:w_crop+target_w]

        x = torch.cat([concat_x, x], dim=1)
        x = self.convblock(x)
        return x