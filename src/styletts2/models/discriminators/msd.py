import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm

LRELU_SLOPE = 0.1


def stft(x, fft_size, hop_size, win_length, window):
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    return torch.abs(x_stft).transpose(2, 1)


class SpecDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        use_spectral_norm=False,
    ):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.discriminators = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                ),
            ]
        )
        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):
        fmap = []
        y = y.squeeze(1)
        y = stft(
            y, self.fft_size, self.shift_size, self.win_length, self.window.to(y.device)
        )
        y = y.unsqueeze(1)
        for d in self.discriminators:
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)
        y = self.out(y)
        fmap.append(y)
        return torch.flatten(y, 1, -1), fmap


class MultiResSpecDiscriminator(nn.Module):
    def __init__(
        self,
        fft_sizes=None,
        hop_sizes=None,
        win_lengths=None,
        window="hann_window",
    ):
        if win_lengths is None:
            win_lengths = [600, 1200, 240]
        if hop_sizes is None:
            hop_sizes = [120, 240, 50]
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 512]
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
                SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
                SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
