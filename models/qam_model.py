import torch.nn as nn

from utils.qam_modem_torch import QAMModulator, QAMDemodulator
from models.channellayer import RayleighChannel, AWGNChannel

class QAMModem(nn.Module):
    def __init__(self, snrdB=25, order=256, channel='Rayleigh'):
        super(QAMModem, self).__init__()
        self.encoder = QAMModulator(order=order)
        self.decoder = QAMDemodulator(order=order)
        if channel.lower() == 'rayleigh':
            self.channel = RayleighChannel(snrdB=snrdB)
        else:
            self.channel = AWGNChannel(snrdB=snrdB)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.channel(x)
        x = self.decoder(x)
        return x