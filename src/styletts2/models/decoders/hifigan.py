import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

from styletts2.models.decoders.common import (
    AdainResBlk1d,
    AdaINResBlock,
    SourceModuleHnNSF,
    init_weights,
)


class Generator(nn.Module):
    def __init__(
        self,
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=np.prod(upsample_rates),
            harmonic_num=8,
            voiced_threshold=10,
        )
        self.f0_upsamp = nn.Upsample(scale_factor=np.prod(upsample_rates))

        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.alphas = nn.ParameterList()

        self.alphas.append(nn.Parameter(torch.ones(1, upsample_initial_channel, 1)))

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_in = upsample_initial_channel // (2**i)
            c_out = upsample_initial_channel // (2 ** (i + 1))

            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        c_in,
                        c_out,
                        k,
                        u,
                        padding=(u // 2 + u % 2),
                        output_padding=u % 2,
                    )
                )
            )

            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        c_out,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(AdaINResBlock(c_out, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(1, c_out, kernel_size=1))
                self.noise_res.append(AdaINResBlock(c_out, 11, [1, 3, 5], style_dim))

            self.alphas.append(nn.Parameter(torch.ones(1, c_out, 1)))
            for _j, (rk, rd) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(AdaINResBlock(c_out, rk, rd, style_dim))

        self.conv_post = weight_norm(nn.Conv1d(c_out, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, s, f0):
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(1, 2)

        for i in range(self.num_upsamples):
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x_source = self.noise_convs[i](har_source)
            x_source = self.noise_res[i](x_source, s)

            x = self.ups[i](x)
            x = x + x_source

            xs = self.resblocks[i * self.num_kernels](x, s)
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels

        x = x + (1 / self.alphas[i + 1]) * (torch.sin(self.alphas[i + 1] * x) ** 2)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            if isinstance(layer, AdaINResBlock):
                layer.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class HiFiGANDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=1024,
        dim_in=512,
        style_dim=64,
        resblock_kernel_sizes=None,
        upsample_rates=None,
        upsample_initial_channel=512,
        resblock_dilation_sizes=None,
        upsample_kernel_sizes=None,
    ):
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [20, 10, 6, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if upsample_rates is None:
            upsample_rates = [10, 5, 3, 2]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, hidden_dim, style_dim)
        self.decode = nn.ModuleList(
            [
                AdainResBlk1d(hidden_dim + 2 + 64, hidden_dim, style_dim),
                AdainResBlk1d(hidden_dim + 2 + 64, hidden_dim, style_dim),
                AdainResBlk1d(hidden_dim + 2 + 64, hidden_dim, style_dim),
                AdainResBlk1d(hidden_dim + 2 + 64, 512, style_dim, upsample=True),
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
