import logging
import random
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monotonic_align import mask_from_lens

from styletts2.config import StyleTTS2Config
from styletts2.models.diffusion.k_diffusion import ADPM2Sampler, DiffusionSampler
from styletts2.models.diffusion.sampler import KarrasSchedule
from styletts2.models.losses import (
    DiscriminatorLoss,
    GeneratorLoss,
    MultiResolutionSTFTLoss,
    WavLMLoss,
)
from styletts2.models.slm_loss import SLMAdversarialLoss
from styletts2.utils.helpers import length_to_mask, maximum_path

_logger = logging.getLogger(__name__)


def log_norm(x, mean=-4, std=4, dim=2):
    return torch.log(torch.exp(x * std + mean).norm(dim=dim))

class StyleTTS2LightningModule(pl.LightningModule):
    def __init__(self, config: StyleTTS2Config, model):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.config = config
        self.model = model

        # We'll use manual optimization because of the complex GAN setup and multiple optimizers
        self.automatic_optimization = False

        # Losses
        self.gl = GeneratorLoss(model.mpd, model.msd)
        self.dl = DiscriminatorLoss(model.mpd, model.msd)
        self.wl = WavLMLoss(
            config.model_params.slm.model,
            model.wd,
            config.preprocess_params.sr,
            config.model_params.slm.sr,
        )
        self.stft_loss = MultiResolutionSTFTLoss()

        # Freeze the WavLM backbone — only the wd discriminator head trains.
        # Training the full WavLM backbone offers negligible TTS benefit and
        # wastes VRAM / compute.
        for p in self.wl.wavlm.parameters():
            p.requires_grad_(False)

        self.sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        self.slmadv = SLMAdversarialLoss(
            model,
            self.wl,
            self.sampler,
            min_len=config.slmadv_params.min_len,
            max_len=config.slmadv_params.max_len,
            batch_percentage=config.slmadv_params.batch_percentage,
            skip_update=config.slmadv_params.iter,
            sig=config.slmadv_params.sig,
        )

    def configure_optimizers(self):
        params = self.config.optimizer_params

        opt_gen = torch.optim.AdamW(
            [
                {'params': self.model.bert.parameters(), 'lr': params.bert_lr * 2},
                {'params': self.model.bert_encoder.parameters()},
                {'params': self.model.predictor.parameters()},
                {'params': self.model.decoder.parameters(), 'lr': params.ft_lr * 2},
                {'params': self.model.text_encoder.parameters()},
                {'params': self.model.predictor_encoder.parameters()},
                {'params': self.model.style_encoder.parameters(), 'lr': params.ft_lr * 2},
                {'params': self.model.diffusion.parameters()},
                {'params': self.model.text_aligner.parameters()},
            ],
            lr=params.lr,
            weight_decay=1e-4,
            betas=(0.0, 0.99),
            eps=1e-9
        )

        opt_msd = torch.optim.AdamW(self.model.msd.parameters(), lr=params.lr, weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9)
        opt_mpd = torch.optim.AdamW(self.model.mpd.parameters(), lr=params.lr, weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9)
        opt_wd = torch.optim.AdamW(self.model.wd.parameters(), lr=params.lr, weight_decay=1e-4, betas=(0.0, 0.99), eps=1e-9)

        # Schedulers
        total_steps = int(self.trainer.estimated_stepping_batches)

        sched_gen = torch.optim.lr_scheduler.OneCycleLR(
            opt_gen, max_lr=[params.bert_lr * 2, params.lr, params.lr, params.ft_lr * 2, params.lr, params.lr, params.ft_lr * 2, params.lr, params.lr],
            total_steps=total_steps, pct_start=0.0, div_factor=1, final_div_factor=1
        )

        sched_msd = torch.optim.lr_scheduler.OneCycleLR(
            opt_msd, max_lr=params.lr, total_steps=total_steps, pct_start=0.0, div_factor=1, final_div_factor=1
        )

        sched_mpd = torch.optim.lr_scheduler.OneCycleLR(
            opt_mpd, max_lr=params.lr, total_steps=total_steps, pct_start=0.0, div_factor=1, final_div_factor=1
        )

        sched_wd = torch.optim.lr_scheduler.OneCycleLR(
            opt_wd, max_lr=params.lr, total_steps=total_steps, pct_start=0.0, div_factor=1, final_div_factor=1
        )

        return [opt_gen, opt_msd, opt_mpd, opt_wd], [sched_gen, sched_msd, sched_mpd, sched_wd]

    def training_step(self, batch, batch_idx):
        # Fix unpacking of optimizers
        opts: Any = self.optimizers()
        opt_gen, opt_msd, opt_mpd, opt_wd = opts[0], opts[1], opts[2], opts[3]

        scheds: Any = self.lr_schedulers()
        sched_gen, sched_msd, sched_mpd, sched_wd = scheds[0], scheds[1], scheds[2], scheds[3]

        loss_params = self.config.loss_params

        # Unpack simplified batch (removed ref_texts, ref_lengths)
        (
            texts,
            input_lengths,
            mels,
            mel_input_length,
            ref_mels,
            _labels,
            waves,
        ) = batch

        n_down = int(self.model.text_aligner.n_down)
        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2**n_down)).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            # Reference styles
            if (self.config.model_params.multispeaker and self.current_epoch >= loss_params.diff_epoch) or self.current_epoch >= loss_params.joint_epoch:
                ref_ss = self.model.style_encoder(ref_mels.unsqueeze(1))
                ref_sp = self.model.predictor_encoder(ref_mels.unsqueeze(1))
                ref = torch.cat([ref_ss, ref_sp], dim=1)
            else:
                ref = None

        # Aligner
        try:
            _ppgs, s2s_pred, s2s_attn = self.model.text_aligner(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
        except Exception as exc:
            _logger.warning(
                "text_aligner failed on batch %d (epoch %d): %s — skipping.",
                batch_idx, self.current_epoch, exc,
            )
            # Return a zero-scalar loss so Lightning's grad scaling stays valid.
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        mask_st = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**n_down))
        s2s_attn_mono = maximum_path(s2s_attn, mask_st)

        # Encode
        t_en = self.model.text_encoder(texts, input_lengths, text_mask)
        asr = t_en @ (s2s_attn if bool(random.getrandbits(1)) else s2s_attn_mono)
        d_gt = s2s_attn_mono.sum(dim=-1).detach()

        # Styles
        ss, gs = [], []
        for bib in range(len(mel_input_length)):
            mel = mels[bib, :, : mel_input_length[bib]].unsqueeze(0).unsqueeze(1)
            ss.append(self.model.predictor_encoder(mel))
            gs.append(self.model.style_encoder(mel))
        # Use squeeze(1) — bare squeeze() collapses the batch dim when B=1.
        s_dur = torch.stack(ss).squeeze(1)
        gs = torch.stack(gs).squeeze(1)
        s_trg = torch.cat([gs, s_dur], dim=-1).detach()

        bert_dur = self.model.bert(texts, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        # Diffusion
        loss_diff, loss_sty = torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)
        if self.current_epoch >= loss_params.diff_epoch:
            num_steps = np.random.randint(3, 5)
            if self.config.model_params.multispeaker:
                s_preds = self.sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1).to(self.device),
                    embedding=bert_dur,
                    features=ref,
                    num_steps=num_steps,
                ).squeeze(1)
                loss_diff = self.model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur, features=ref).mean()
            else:
                s_preds = self.sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1).to(self.device),
                    embedding=bert_dur,
                    num_steps=num_steps,
                ).squeeze(1)
                loss_diff = self.model.diffusion(s_trg.unsqueeze(1), embedding=bert_dur).mean()
            loss_sty = F.l1_loss(s_preds, s_trg.detach())

        # Predictor
        d, p = self.model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

        # Prepare clips for decoder
        mel_len = min(int(mel_input_length.min().item() / 2 - 1), self.config.max_len // 2)
        en_clip, p_en_clip, gt_clip, wav_clip = [], [], [], []
        for bib in range(len(mel_input_length)):
            m_len = int(mel_input_length[bib].item() / 2)
            start = np.random.randint(0, m_len - mel_len)
            en_clip.append(asr[bib, :, start : start + mel_len])
            p_en_clip.append(p[bib, :, start : start + mel_len])
            gt_clip.append(mels[bib, :, (start * 2) : ((start + mel_len) * 2)])
            y = waves[bib][(start * 2) * 300 : ((start + mel_len) * 2) * 300]
            wav_clip.append(torch.from_numpy(y).to(self.device))

        wav_clip = torch.stack(wav_clip).float().detach().unsqueeze(1)
        en_clip = torch.stack(en_clip)
        p_en_clip = torch.stack(p_en_clip)
        gt_clip = torch.stack(gt_clip).detach()

        # --- Discriminator Step ---
        opt_msd.zero_grad()
        opt_mpd.zero_grad()

        with torch.no_grad():
            f0_real, _, _ = self.model.pitch_extractor(gt_clip.unsqueeze(1))
            n_real = log_norm(gt_clip.unsqueeze(1)).squeeze(1)
            s_clip = self.model.style_encoder(gt_clip.unsqueeze(1))

        f0_fake, n_fake = self.model.predictor.f0_n_train(p_en_clip, self.model.predictor_encoder(gt_clip.unsqueeze(1)))
        y_rec = self.model.decoder(en_clip, f0_fake, n_fake, s_clip)

        d_loss = self.dl(wav_clip.detach(), y_rec.detach())
        self.manual_backward(d_loss)
        opt_msd.step()
        opt_mpd.step()

        # --- Generator Step ---
        opt_gen.zero_grad()
        loss_mel = self.stft_loss(y_rec, wav_clip)
        loss_gen_all = self.gl(wav_clip, y_rec)
        loss_lm = self.wl(wav_clip.detach().squeeze(), y_rec.squeeze()).mean()

        loss_dur, loss_ce = self._compute_dur_ce_losses(d, d_gt, input_lengths, texts)
        loss_s2s = F.cross_entropy(s2s_pred.flatten(0, 1), texts.flatten())
        loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
        loss_f0_rec = F.smooth_l1_loss(f0_real, f0_fake) / 10
        loss_norm_rec = F.smooth_l1_loss(n_real, n_fake)

        g_loss = (
            loss_params.lambda_mel * loss_mel
            + loss_params.lambda_f0 * loss_f0_rec
            + loss_params.lambda_ce * loss_ce
            + loss_params.lambda_norm * loss_norm_rec
            + loss_params.lambda_dur * loss_dur
            + loss_params.lambda_gen * loss_gen_all
            + loss_params.lambda_slm * loss_lm
            + loss_params.lambda_sty * loss_sty
            + loss_params.lambda_diff * loss_diff
            + loss_params.lambda_mono * loss_mono
            + loss_params.lambda_s2s * loss_s2s
        )

        self.manual_backward(g_loss)

        # --- SLM Adversarial Step ---
        if self.current_epoch >= loss_params.joint_epoch:
            opt_wd.zero_grad()
            slm_out = self.slmadv(
                self.global_step,
                wav_clip.detach(),
                y_rec.detach(),
                waves,
                mel_input_length,
                texts,
                input_lengths,
                ref,
            )

            if slm_out is not None:
                d_slm_loss, g_slm_loss, _ = slm_out
                if d_slm_loss != 0:
                    self.manual_backward(d_slm_loss)
                    opt_wd.step()

                self.manual_backward(g_slm_loss)

                self.log("train/d_slm", d_slm_loss, prog_bar=True)
                self.log("train/g_slm", g_slm_loss, prog_bar=True)

        opt_gen.step()

        # Step schedulers
        sched_gen.step()
        sched_msd.step()
        sched_mpd.step()
        # Only advance WD scheduler once it is actually training to preserve
        # its LR budget for the epochs it matters.
        if self.current_epoch >= loss_params.joint_epoch:
            sched_wd.step()

        # Logging
        self.log("train/mel_loss", loss_mel, prog_bar=True)
        self.log("train/g_loss", g_loss, prog_bar=False)
        self.log("train/d_loss", d_loss, prog_bar=False)
        self.log("train/dur_loss", loss_dur, prog_bar=False)
        self.log("train/f0_loss", loss_f0_rec, prog_bar=False)

        return g_loss

    def _compute_dur_ce_losses(self, d, d_gt, input_lengths, texts):
        loss_ce, loss_dur = 0, 0
        for _pred, _gt, _len in zip(d, d_gt, input_lengths):
            _pred = _pred[:_len, :]
            _gt = _gt[:_len].long()
            _trg = torch.zeros_like(_pred)
            for p in range(_trg.shape[0]):
                _trg[p, : _gt[p]] = 1
            _dur_pred = torch.sigmoid(_pred).sum(dim=1)
            loss_dur += F.l1_loss(_dur_pred[1 : _len - 1], _gt[1 : _len - 1].float())
            loss_ce += F.binary_cross_entropy_with_logits(_pred.flatten(), _trg.flatten())
        return loss_dur / texts.size(0), loss_ce / texts.size(0)

    def validation_step(self, batch, batch_idx):
        (
            texts,
            input_lengths,
            mels,
            mel_input_length,
            ref_mels,
            _labels,
            waves,
        ) = batch

        n_down = int(self.model.text_aligner.n_down)
        mask = length_to_mask(mel_input_length // (2**n_down)).to(self.device)
        text_mask = length_to_mask(input_lengths).to(self.device)

        # Basic validation: compute mel reconstruction loss
        with torch.inference_mode():
            # Styles
            ref_ss = self.model.style_encoder(ref_mels.unsqueeze(1))
            ref_sp = self.model.predictor_encoder(ref_mels.unsqueeze(1))

            # Encoders
            t_en = self.model.text_encoder(texts, input_lengths, text_mask)

            # Predictor
            # For validation we use ground truth alignment if available, or just skip complex parts
            # Here we just want a proxy for convergence
            _ppgs, _s2s_pred, s2s_attn = self.model.text_aligner(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
            mask_st = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_st)

            # Reconstruct
            en = (t_en @ s2s_attn_mono)
            f0_fake, n_fake = self.model.predictor.f0_n_train(en, ref_sp)
            y_rec = self.model.decoder(en, f0_fake, n_fake, ref_ss)

            # Mel loss
            # Handle varying wave lengths for validation
            max_wave_len = max([w.shape[0] for w in waves])
            padded_waves = []
            for w in waves:
                pad_w = np.pad(w, (0, max_wave_len - w.shape[0]))
                padded_waves.append(pad_w)

            wave_tensor = torch.from_numpy(np.stack(padded_waves)).float().to(self.device).unsqueeze(1)
            loss_mel = self.stft_loss(y_rec, wave_tensor)

            self.log("val/mel_loss", loss_mel, sync_dist=True, prog_bar=True)

        return loss_mel
