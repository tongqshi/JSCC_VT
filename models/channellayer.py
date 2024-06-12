import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class RayleighChannel(nn.Module):
    def __init__(self, snrdB=None):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) if snrdB is not None else None

    def forward(self, inputs):
        # power normalization
        normalizer = torch.sqrt(torch.mean(inputs ** 2))
        x = inputs / normalizer

        h = torch.normal(mean=0, std=1.0, size=inputs.shape, requires_grad=True, device='cuda')

        n = torch.normal(mean=0, std=torch.sqrt(torch.tensor(1 / self.snr)), size=inputs.shape, requires_grad=True, device='cuda')

        y = h * x + n

        yhat = y * normalizer
        yhat = torch.where(h != 0, yhat / h, torch.zeros_like(yhat))  # Avoid division by zero


        return yhat


class AWGNChannel(nn.Module):
    def __init__(self, snrdB=None):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) if snrdB is not None else None

    def forward(self, inputs):
        # power normalization
        normalizer = torch.sqrt(torch.mean(inputs ** 2))
        x = inputs / normalizer

        # snrdB = random.randint(10, 40)
        # snr = 10 ** (snrdB / 10)

        n = torch.normal(mean=0, std=torch.sqrt(torch.tensor(1 / self.snr)), size=inputs.shape, requires_grad=True, device='cuda')
        self.n = n
        y = x + n

        yhat = y * normalizer
        return yhat

