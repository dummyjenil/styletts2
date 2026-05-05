import math

import torch
import torch.nn.functional as F
import torchaudio.functional as audio_F
from torch import nn
from transformers import AlbertModel


class CustomAlbert(AlbertModel):
    def __init__(self, config):
        super().__init__(config, False)
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state


def init_weights(m, gain="linear"):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(gain))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MFCC(nn.Module):
    def __init__(self, n_mfcc=40, n_mels=80):
        super().__init__()
        self.register_buffer("dct_mat", audio_F.create_dct(n_mfcc, n_mels, "ortho"))

    def forward(self, x):
        if x.ndim == 2:
            return x.T.matmul(self.dct_mat).T
        return x.transpose(1, 2).matmul(self.dct_mat).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, q_dim, m_dim, a_dim, n_filters, kernel_size):
        super().__init__()
        self.q_layer = nn.Linear(q_dim, a_dim, bias=False)
        self.m_layer = nn.Linear(m_dim, a_dim, bias=False)
        self.v = nn.Linear(a_dim, 1, bias=False)
        self.loc_conv = nn.Conv1d(
            2, n_filters, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.loc_dense = nn.Linear(n_filters, a_dim, bias=False)
        for layer in [self.q_layer, self.m_layer, self.loc_dense]:
            init_weights(layer, "tanh")
        init_weights(self.v)

    def forward(self, q, m, processed_m, w_cat, mask):
        energies = self.v(
            torch.tanh(
                self.q_layer(q).unsqueeze(1)
                + self.loc_dense(self.loc_conv(w_cat).transpose(1, 2))
                + processed_m
            )
        ).squeeze(-1)
        if mask is not None:
            energies.masked_fill_(mask, -float("inf"))
        w = F.softmax(energies, dim=1)
        return torch.bmm(w.unsqueeze(1), m).squeeze(1), w


class ConvBlock(nn.Module):
    def __init__(self, dim, n_conv=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, 3, padding=3**i, dilation=3**i),
                    nn.ReLU(),
                    nn.GroupNorm(8, dim),
                    nn.Dropout(0.2),
                    nn.Conv1d(dim, dim, 3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )
                for i in range(n_conv)
            ]
        )
        for b in self.blocks:
            b.apply(init_weights)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class ASRS2S(nn.Module):
    def __init__(
        self, embed_dim=256, hidden_dim=512, n_filters=32, kernel_size=63, n_token=40
    ):
        super().__init__()
        self.hidden_dim, self.sos, self.eos, self.unk = hidden_dim, 1, 2, 3
        self.embed = nn.Embedding(n_token, embed_dim)
        nn.init.uniform_(
            self.embed.weight, -math.sqrt(6 / hidden_dim), math.sqrt(6 / hidden_dim)
        )

        self.attention = Attention(
            hidden_dim, hidden_dim, hidden_dim, n_filters, kernel_size
        )
        self.decoder_rnn = nn.LSTMCell(hidden_dim + embed_dim, hidden_dim)
        self.proj_hidden = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh()
        )
        self.proj_tokens = nn.Linear(hidden_dim, n_token)
        for m in [self.proj_hidden[0], self.proj_tokens]:
            init_weights(m)

    def forward(self, memory, mask, text_input):
        batch_size, seq_len, hidden_dim = memory.shape
        device = memory.device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        w = torch.zeros(batch_size, seq_len, device=device)
        w_cum = torch.zeros(batch_size, seq_len, device=device)
        context = torch.zeros(batch_size, hidden_dim, device=device)

        processed_m = self.attention.m_layer(memory)
        text_masked = text_input.clone().masked_fill_(
            torch.rand_like(text_input, dtype=torch.float) < 0.1, self.unk
        )
        inputs = self.embed(
            torch.cat(
                [
                    torch.full(
                        (batch_size, 1), self.sos, device=device, dtype=torch.long
                    ),
                    text_masked,
                ],
                1,
            )
        ).transpose(0, 1)

        h_outs, l_outs, w_outs = [], [], []
        for x in inputs:
            h, c = self.decoder_rnn(torch.cat([x, context], -1), (h, c))
            context, w = self.attention(
                h, memory, processed_m, torch.stack([w, w_cum], 1), mask
            )
            w_cum += w
            h_proj = self.proj_hidden(torch.cat([h, context], -1))
            logit = self.proj_tokens(F.dropout(h_proj, 0.5, self.training))
            h_outs.append(h_proj)
            l_outs.append(logit)
            w_outs.append(w)

        return torch.stack(h_outs, 1), torch.stack(l_outs, 1), torch.stack(w_outs, 1)


class ASRCNN(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, n_token=35, n_layers=6):
        super().__init__()
        self.to_mfcc = MFCC()
        self.init_cnn = nn.Conv1d(input_dim // 2, hidden_dim, 7, stride=2, padding=3)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(ConvBlock(hidden_dim), nn.GroupNorm(1, hidden_dim))
                for _ in range(n_layers)
            ]
        )
        self.proj = nn.Conv1d(hidden_dim, hidden_dim // 2, 1)
        self.ctc = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_token),
        )
        self.s2s = ASRS2S(hidden_dim=hidden_dim // 2, n_token=n_token)
        self.n_down = 1
        for m in [self.init_cnn, self.proj, self.ctc[0], self.ctc[2]]:
            init_weights(m)

    def forward(self, x, mask=None, text=None):
        x = self.init_cnn(self.to_mfcc(x))
        for layer in self.layers:
            x = layer(x)
        feat = self.proj(x).transpose(1, 2)
        ctc_logit = self.ctc(feat)
        if text is None:
            return ctc_logit
        _, s2s_logit, s2s_attn = self.s2s(feat, mask, text)
        return ctc_logit, s2s_logit, s2s_attn

    def get_feature(self, x):
        x = self.init_cnn(self.to_mfcc(x.squeeze(1) if x.ndim == 3 else x))
        for layer in self.layers:
            x = layer(x)
        return self.proj(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )
        self.conv1by1 = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if self.downsample
            else None
        )

    def forward(self, x):
        x = self.pre_conv(x)
        return self.conv(x) + self.conv1by1(x) if self.conv1by1 else x


class JDCNet(nn.Module):
    def __init__(self, num_class=722, seq_len=31, leaky_relu_slope=0.01):
        super().__init__()
        self.num_class = num_class
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
        )
        self.res_block1 = ResBlock(64, 128)
        self.res_block2 = ResBlock(128, 192)
        self.res_block3 = ResBlock(192, 256)
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.2),
        )
        self.bilstm_classifier = nn.LSTM(
            input_size=512, hidden_size=256, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(512, num_class)
        self.apply(self.init_weights)

    def forward(self, x):
        seq_len = x.shape[-1]
        x = x.float().transpose(-1, -2)

        x = self.conv_block(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.pool_block[0](x)
        x = self.pool_block[1](x)
        gan_feature = x.transpose(-1, -2)
        x = self.pool_block[2](x)

        # (b, 256, 31, 2) => (b, 31, 256, 2) => (b, 31, 512)
        x = x.permute(0, 2, 1, 3).contiguous().view((-1, seq_len, 512))
        x, _ = self.bilstm_classifier(x)
        x = self.classifier(x.contiguous().view((-1, 512)))
        return (
            x.view((-1, seq_len, self.num_class)),
            gan_feature,
            None,
        )  # simplified return

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
