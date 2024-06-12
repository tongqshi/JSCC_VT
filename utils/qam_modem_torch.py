import torch
import torch.nn as nn
import math
import time
import numpy as np

MAX_ORDER = 256

class QAMModulator(nn.Module):
    def __init__(self, order=256):
        super(QAMModulator, self).__init__()
        assert order <= MAX_ORDER, f"Order should not be greater than {MAX_ORDER}"
        self.order = order
        self.n_bit = int(math.log2(self.order))

    def forward(self, inputs):
        assert inputs.dtype in [torch.int8, torch.int16, torch.int32, torch.int64], \
               "input tensor dtype should be int"
        avg_power = math.sqrt((self.order - 1) / 3 * 2)

        # bit split
        lower_half_bit_mask = (1 << (self.n_bit // 2)) - 1
        upper_half_bit_mask = lower_half_bit_mask << (self.n_bit // 2)

        lower_bit = inputs & lower_half_bit_mask
        upper_bit = (inputs & upper_half_bit_mask) >> (self.n_bit // 2)

        output = torch.stack([upper_bit, lower_bit])

        # to gray code
        output = (output ^ (output >> 1)) + 1

        # center to zero and power normalization
        output = output.float()
        output = (2 * output - 1 - math.sqrt(self.order)) / avg_power

        return output


class QAMDemodulator(nn.Module):
    def __init__(self, order=256):
        super(QAMDemodulator, self).__init__()
        assert order <= MAX_ORDER, f"Order should not be greater than {MAX_ORDER}"
        self.order = order
        self.n_bit = int(math.log2(self.order))

    def forward(self, inputs):
        assert inputs.shape[0] == 2, \
               "first dimension size of the given tensor should be 2 (for in-phase and quadrature-phase, respectively)"

        avg_power = math.sqrt((self.order - 1) / 3 * 2)

        # QAM detection
        yhat = (inputs * avg_power / 2).floor().int() * 2 + 1
        max_val = 2 * (1 << (self.n_bit // 2)) - 1 - int(math.sqrt(self.order))
        min_val = 1 - int(math.sqrt(self.order))
        yhat = torch.clamp(yhat, min_val, max_val)

        # undo zero-centering
        yhat = ((yhat + int(math.sqrt(self.order)) + 1) // 2 - 1).short()

        # gray code to binary
        for shift in [128, 64, 32, 16, 8, 4, 2, 1]:
            yhat = yhat ^ (yhat >> shift)

        # bit concatenation
        output = (yhat[0] << (self.n_bit // 2)) | yhat[1]

        return output


if __name__ == '__main__':
    start = time.time()

    m = 256
    n_bit = int(math.log2(m))
    snrdB = 15
    num_repeat = int(1e+6)

    mod = QAMModulator(order=m)
    demod = QAMDemodulator(order=m)

    snr = 10 ** (snrdB / 10)  # in dB
    sigma = 1 / math.sqrt(snr * 2)
    biterror = 0

    source = torch.randint(low=0, high=255, size=(num_repeat,), dtype=torch.int16)
    x = mod(source)

    noise = torch.normal(mean=0, std=sigma, size=(2, num_repeat)).float()
    y = x + noise
    shat = demod(y)

    end = time.time() - start
    print(f'N: {num_repeat}, Elapsed: {end:.4f}s')

    power = torch.mean(x ** 2)
    print(f'SNR: {1 / (2 * sigma ** 2):.4f} / Target: {snr:.4f}')
    print(f'AVG Power: {power:.4f}')
    print(f'Eb/N0: {power / (2 * sigma ** 2):.4f}')

    source_output_xor = source ^ shat
    for i in source_output_xor:
        biterror += bin(i).count("1")
    print(f'BER: {biterror / num_repeat / n_bit}')
