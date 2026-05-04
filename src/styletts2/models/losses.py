import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from transformers import AutoModel


class SpectralConvergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)


class STFTLoss(nn.Module):
    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window
    ):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=shift_size,
            window_fn=window,
        )
        self.spectral_convergence_loss = SpectralConvergenceLoss()

    def forward(self, x, y):
        x_mag = self.to_mel(x)
        y_mag = self.to_mel(y)

        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std

        return self.spectral_convergence_loss(x_mag, y_mag)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=None,
        hop_sizes=None,
        win_lengths=None,
        window=torch.hann_window,
    ):
        if win_lengths is None:
            win_lengths = [600, 1200, 240]
        if hop_sizes is None:
            hop_sizes = [120, 240, 50]
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 512]
        super().__init__()
        self.stft_losses = nn.ModuleList(
            [
                STFTLoss(fs, ss, wl, window)
                for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )

    def forward(self, x, y):
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_loss += f(x, y)
        return sc_loss / len(self.stft_losses)


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        loss_val = torch.mean((1 - dg) ** 2)
        gen_losses.append(loss_val)
        loss += loss_val
    return loss, gen_losses


def tprls_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_dg = torch.median(dr - dg)
        l_rel = torch.mean((((dr - dg) - m_dg) ** 2)[dr < dg + m_dg])
        loss += tau - F.relu(tau - l_rel)
    return loss


class GeneratorLoss(nn.Module):
    def __init__(self, mpd, msd):
        super().__init__()
        self.mpd = mpd
        self.msd = msd

    def forward(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)

        loss_fm = feature_loss(fmap_f_r, fmap_f_g) + feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_rel = tprls_loss(y_df_hat_r, y_df_hat_g) + tprls_loss(
            y_ds_hat_r, y_ds_hat_g
        )

        return (loss_gen_s + loss_gen_f + loss_fm + loss_rel).mean()


class DiscriminatorLoss(nn.Module):
    def __init__(self, mpd, msd):
        super().__init__()
        self.mpd = mpd
        self.msd = msd

    def forward(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_rel = tprls_loss(y_df_hat_r, y_df_hat_g) + tprls_loss(
            y_ds_hat_r, y_ds_hat_g
        )

        return (loss_disc_s + loss_disc_f + loss_rel).mean()


class WavLMLoss(nn.Module):
    def __init__(self, model_name, wd, sr, slm_sr=16000):
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(model_name)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(sr, slm_sr)

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states

        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16.squeeze(), output_hidden_states=True
        ).hidden_states

        floss = torch.tensor(0.0).to(wav.device)
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))
        return floss.mean()

    def generator(self, y_rec):
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, output_hidden_states=True
        ).hidden_states
        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings)
        return torch.mean((1 - y_df_hat_g) ** 2)

    def discriminator(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(
                input_values=y_rec_16, output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_df_hat_r = self.wd(y_embeddings)
        y_df_hat_g = self.wd(y_rec_embeddings)

        return (torch.mean((1 - y_df_hat_r) ** 2) + torch.mean(y_df_hat_g**2)).mean()


