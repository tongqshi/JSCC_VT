import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import random

from models.resblock import ResBlock
from models.vitencoder import ViTEncoder
from utils.qam_modem_torch import QAMModulator, QAMDemodulator
from models.channellayer import RayleighChannel, AWGNChannel
from models.unet_layer import Attention_UNet

class E2EImageCommunicator(nn.Module):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super(E2EImageCommunicator, self).__init__()

        self.encoder = nn.Sequential(
            ResBlock(3, 32, stride=1, is_bottleneck=True),
            ResBlock(128, 64, stride=1, is_bottleneck=True),
            ResBlock(256, 64, stride=1, is_bottleneck=True),
            ResBlock(256, 32, stride=1, is_bottleneck=True),
        )

        self.conv1 = nn.Conv2d(128, 3, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

        self.qammod = QAMModulator(order=256)
        self.qamdemod = QAMDemodulator(order=256)

        if channel.lower() == 'rayleigh':
            self.channel = RayleighChannel(snrdB=snrdB)
        else:
            self.channel = AWGNChannel(snrdB=snrdB)

        self.decoder = nn.Sequential(
            *[ViTEncoder(num_heads=4, head_size=64, mlp_dim=[3, 128, 3]) for _ in range(l)]
        )

        self.residual_proj = nn.Conv2d(3, 3, kernel_size=1, stride=1)

        KERNEL_SIZE = 3

        # self.autoencoder = nn.Sequential(
        #     nn.Conv2d(3, filters[0], KERNEL_SIZE, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(filters[0], filters[1], KERNEL_SIZE, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(filters[1], filters[2], 1, padding=0),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(filters[2], filters[1], KERNEL_SIZE, padding=1, stride=2, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(filters[1], 3, KERNEL_SIZE, padding=1, stride=2, output_padding=1),
        #     nn.ReLU()
        # )

        self.autoencoder = Attention_UNet(in_channels=3)

        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, stride=1)

        self.mhsa = nn.MultiheadAttention(embed_dim=3*8*8, num_heads=4, batch_first=True)

    def forward(self, inputs):
        x = self.encoder(inputs)

        x = self.sigmoid(self.conv1(x))

        x = self.channel(x)

        x = self.decoder(x.detach())
        x1 = self.residual_proj(x)
        x = self.autoencoder(x)

        x = rearrange(x, 'b c (p1 h) (p2 w) ->b (p1 p2) (c h w)', h=8, w=8)
        x1 = rearrange(x1, 'b c (p1 h) (p2 w) ->b (p1 p2) (c h w)', h=8, w=8)
        x1, _ = self.mhsa(x, x1, x1)
        x = rearrange(x, 'b (p1 p2) (c h w) ->b c (p1 h) (p2 w)', c=3, p1=4, h=8)
        x1 = rearrange(x1, 'b (p1 p2) (c h w) ->b c (p1 h) (p2 w)', c=3, p1=4, h=8)

        x = x + x1
        x = self.sigmoid(self.conv2(x))

        return x



class E2E_Encoder(E2EImageCommunicator):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super().__init__(filters, l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.sigmoid(self.conv1(x))

        return x


class E2E_Channel(E2EImageCommunicator):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super().__init__(filters, l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.sigmoid(self.conv1(x))

        x = self.channel(x)

        return x


class E2E_Decoder(E2EImageCommunicator):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super().__init__(filters, l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.sigmoid(self.conv1(x))

        x = self.channel(x)

        x = self.decoder(x)

        return x


class E2E_AutoEncoder(E2EImageCommunicator):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super().__init__(filters, l, snrdB, channel)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.sigmoid(self.conv1(x))

        x = self.channel(x)

        x = self.decoder(x)
        x1 = self.residual_proj(x)
        x = self.autoencoder(x)

        x = rearrange(x, 'b c (p1 h) (p2 w) ->b (p1 p2) (c h w)', h=8, w=8)
        x1 = rearrange(x1, 'b c (p1 h) (p2 w) ->b (p1 p2) (c h w)', h=8, w=8)
        x1, _ = self.mhsa(x, x1, x1)
        x = rearrange(x, 'b (p1 p2) (c h w) ->b c (p1 h) (p2 w)', c=3, p1=4, h=8)
        x1 = rearrange(x1, 'b (p1 p2) (c h w) ->b c (p1 h) (p2 w)', c=3, p1=4, h=8)

        x = x + x1
        x = self.sigmoid(self.conv2(x))
        return x


class E2E_Decoder_Network(E2EImageCommunicator):
    def __init__(self, filters=[32, 64, 128], l=4, snrdB=25, channel='Rayleigh'):
        super().__init__(filters, l, snrdB, channel)

    def call(self, inputs):
        x = self.decoder(inputs)
        x1 = self.residual_proj(x)
        x = self.autoencoder(x)

        x = rearrange(x, 'b c (p1 h) (p2 w) ->b (p1 p2) (c h w)', h=8, w=8)
        x1 = rearrange(x1, 'b c (p1 h) (p2 w) ->b (p1 p2) (c h w)', h=8, w=8)
        x1, _ = self.mhsa(x, x1, x1)
        x = rearrange(x, 'b (p1 p2) (c h w) ->b c (p1 h) (p2 w)', c=3, p1=4, h=8)
        x1 = rearrange(x1, 'b (p1 p2) (c h w) ->b c (p1 h) (p2 w)', c=3, p1=4, h=8)

        x = x + x1
        x = self.sigmoid(self.conv2(x))

        return x

