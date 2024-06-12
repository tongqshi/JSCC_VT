import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention_unet_blocks import ConvBlock, ContractingBlock, ExpansiveBlock
# from attention_unet_blocks import ConvBlock, ContractingBlock, ExpansiveBlock


class Attention_UNet(nn.Module):
    def __init__(self, in_channels, filters=[16, 32, 64, 128, 256]):
        super(Attention_UNet, self).__init__()
        assert len(filters) > 0, "filters should have at least one element"

        conv_filters = filters[0]
        self.convblock1 = ConvBlock(in_channels, [conv_filters, conv_filters])

        self.contract_layers = nn.ModuleList()
        contracting_path_filters = filters[1:]
        in_channels = conv_filters
        for num_channel in contracting_path_filters:
            self.contract_layers.append(ContractingBlock(in_channels, [num_channel, num_channel]))
            in_channels = num_channel


        self.expand_layers = nn.ModuleList()
        expanding_path_filters = list(filters[::-1][1:])
        for num_channel in expanding_path_filters:
            self.expand_layers.append(ExpansiveBlock(in_channels, [num_channel, num_channel, num_channel]))
            in_channels = num_channel

        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.convblock1(inputs)
        residuals = []

        for contracting_block in self.contract_layers:
            residuals.append(x)
            x = contracting_block(x)


        for expanding_block in self.expand_layers:
            r = residuals.pop()
            x = expanding_block(x, r)


        output = self.sigmoid(self.conv(x))
        return output
