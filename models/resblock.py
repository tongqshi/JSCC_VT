import torch.nn as nn
import torch

class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, is_bottleneck=True):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu1 = nn.ReLU(inplace=True)

        if is_bottleneck:
            self.relu2 = nn.ReLU(inplace=True)
            self.conv_3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1,padding=0)
            self.bn3 = nn.BatchNorm2d(out_channels*4)
        else:
            self.relu2 = EmptyLayer()
            self.conv_3 = EmptyLayer()
            self.bn3 = EmptyLayer()

        ds_out_channels = out_channels * 4 if is_bottleneck else out_channels
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, ds_out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(ds_out_channels)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        residual = self.projection(x)
        out = self.conv_1(x)
        out = self.bn1(out)
        # print(self.conv_1.weight)
        # print(self.conv_1.weight.grad)

        out = self.relu1(out)
        out = self.conv_2(out)
        out = self.bn2(out)

        out = self.relu2(out)
        out = self.conv_3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu1(out)

        return out
