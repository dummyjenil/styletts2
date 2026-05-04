from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import get_window
from torch import Tensor, nn
from torch.nn.utils import weight_norm

from styletts2.models.decoders.common import (
    LRELU_SLOPE,
    AdainResBlk1d,
    AdaINResBlock,
    SourceModuleHnNSF,
    init_weights,
)


class TorchSTFT(nn.Module):
    def __init__(
        self, filter_length=800, hop_length=200, win_length=800, window="hann"
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer(
            "window",
            torch.from_numpy(
                get_window(window, win_length, fftbins=True).astype(np.float32)
            ),
        )

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=cast(Tensor, self.window),
            return_complex=True,
        )
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=cast(Tensor, self.window),
        )
        return inverse_transform.unsqueeze(-2)


class Generator(nn.Module):
    def __init__(
        self,
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=np.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num=8,
            voiced_threshold=10,
        )
        self.f0_upsamp = nn.Upsample(
            scale_factor=np.prod(upsample_rates) * gen_istft_hop_size
        )
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_in = upsample_initial_channel // (2**i)
            c_out = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(c_in, c_out, k, u, padding=(k - u) // 2))
            )
            for _j, (rk, rd) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(AdaINResBlock(c_out, rk, rd, style_dim))

            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        gen_istft_n_fft + 2,
                        c_out,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(AdaINResBlock(c_out, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(
                    nn.Conv1d(gen_istft_n_fft + 2, c_out, kernel_size=1)
                )
                self.noise_res.append(AdaINResBlock(c_out, 11, [1, 3, 5], style_dim))

        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(
            nn.Conv1d(c_out, self.post_n_fft + 2, 7, 1, padding=3)
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )

    def forward(self, x, s, f0):
        with torch.no_grad():
            f0_up = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, _, _ = self.m_source(f0_up)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = self.resblocks[i * self.num_kernels](x, s)
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)


class ISTFTDecoder(nn.Module):
    def __init__(
        self,
        dim_in=512,
        f0_channel=512,
        style_dim=64,
        dim_out=80,
        resblock_kernel_sizes=None,
        upsample_rates=None,
        upsample_initial_channel=512,
        resblock_dilation_sizes=None,
        upsample_kernel_sizes=None,
        gen_istft_n_fft=20,
        gen_istft_hop_size=5,
    ):
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [20, 12]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if upsample_rates is None:
            upsample_rates = [10, 6]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList(
            [
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True),
            ]
        )
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, 3, 2, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, 3, 2, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(512, 64, 1)))
        self.generator = Generator(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
        )

    def forward(self, asr, f0_curve, n, s):
        f0 = self.F0_conv(f0_curve.unsqueeze(1))
        n = self.N_conv(n.unsqueeze(1))
        x = self.encode(torch.cat([asr, f0, n], dim=1), s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, f0, n], dim=1)
            x = block(x, s)
            if block.upsample_type:
                res = False
        return self.generator(x, s, f0_curve)
