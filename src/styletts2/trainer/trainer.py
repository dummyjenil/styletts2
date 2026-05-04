import logging
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from monotonic_align import mask_from_lens
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
from styletts2.models.styletts2 import StyleTTS2Model
from styletts2.trainer.optimizer import build_optimizer
from styletts2.utils.helpers import length_to_mask, maximum_path

logger = logging.getLogger(__name__)


def log_norm(x, mean=-4, std=4, dim=2):
    return torch.log(torch.exp(x * std + mean).norm(dim=dim))


class StyleTTS2Trainer:
    def __init__(
        self,
        config: StyleTTS2Config,
        model: StyleTTS2Model,
        train_dataloader,
        val_dataloader,
        device: str = "cuda",
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        self.model.to(device)

        # Losses
        self.gl = GeneratorLoss(model.mpd, model.msd).to(device)
        self.dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
        self.wl = WavLMLoss(
            config.model_params.slm.model,
            model.wd,
            config.preprocess_params.sr,
            config.model_params.slm.sr,
        ).to(device)
        self.stft_loss = MultiResolutionSTFTLoss().to(device)

        self.sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        # Optimizer
        self.optimizer = self._build_optimizer()

        # SLM Adversarial Loss

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

        self.writer = SummaryWriter(osp.join(config.log_dir, "tensorboard"))
        self.iters = 0
        self.start_epoch = 0
        self.best_loss = float("inf")

    def _build_optimizer(self):
        params = self.config.optimizer_params
        # Access submodules of StyleTTS2Model
        model_params = {
            "bert": self.model.bert.parameters(),
            "bert_encoder": self.model.bert_encoder.parameters(),
            "predictor": self.model.predictor.parameters(),
            "decoder": self.model.decoder.parameters(),
            "text_encoder": self.model.text_encoder.parameters(),
            "predictor_encoder": self.model.predictor_encoder.parameters(),
            "style_encoder": self.model.style_encoder.parameters(),
            "diffusion": self.model.diffusion.parameters(),
            "text_aligner": self.model.text_aligner.parameters(),
            "pitch_extractor": self.model.pitch_extractor.parameters(),
            "mpd": self.model.mpd.parameters(),
            "msd": self.model.msd.parameters(),
            "wd": self.model.wd.parameters(),
        }

        scheduler_params = {
            "max_lr": params.lr,
            "pct_start": 0.0,
            "epochs": self.config.epochs,
            "steps_per_epoch": len(self.train_dataloader),
        }

        scheduler_dict = {key: scheduler_params.copy() for key in model_params}
        scheduler_dict["bert"]["max_lr"] = params.bert_lr * 2
        scheduler_dict["decoder"]["max_lr"] = params.ft_lr * 2
        scheduler_dict["style_encoder"]["max_lr"] = params.ft_lr * 2

        return build_optimizer(model_params, scheduler_dict, params.lr)

    def train(self):
        config = self.config
        for epoch in range(self.start_epoch, config.epochs):
            self.model.train()
            epoch_start_time = time.time()

            pbar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}"
            )
            for i, batch in enumerate(pbar):
                loss_dict = self._train_step(batch, epoch, i)
                self.iters += 1

                if (i + 1) % config.log_interval == 0:
                    pbar.set_postfix(mel_loss=f"{loss_dict['mel_loss']:.4f}")
                    self._log_to_tensorboard(loss_dict, "train")

            self._validate(epoch)
            self._save_checkpoint(epoch)
            logger.info(f"Epoch {epoch + 1} took {time.time() - epoch_start_time:.2f}s")

    def _train_step(self, batch, epoch, step):  # noqa: PLR0912
        model = self.model
        device = self.device
        loss_params = self.config.loss_params
        loss_dict = {}

        waves = batch[0]
        batch = [b.to(device) for b in batch[1:]]
        (
            texts,
            input_lengths,
            _ref_texts,
            _ref_lengths,
            mels,
            mel_input_length,
            ref_mels,
        ) = batch

        n_down = int(model.text_aligner.n_down)
        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2**n_down)).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            # Reference styles
            if (
                self.config.model_params.multispeaker
                and epoch >= loss_params.diff_epoch
            ) or epoch >= loss_params.joint_epoch:
                ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                ref = torch.cat([ref_ss, ref_sp], dim=1)
            else:
                ref = None

        # Aligner
        try:
            _ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
        except Exception:
            return {"mel_loss": 0.0}

        mask_st = mask_from_lens(
            s2s_attn, input_lengths, mel_input_length // (2**n_down)
        )
        s2s_attn_mono = maximum_path(s2s_attn, mask_st)

        # Encode
        t_en = model.text_encoder(texts, input_lengths, text_mask)
        asr = t_en @ (s2s_attn if bool(random.getrandbits(1)) else s2s_attn_mono)
        d_gt = s2s_attn_mono.sum(dim=-1).detach()

        # Styles
        ss, gs = [], []
        for bib in range(len(mel_input_length)):
            mel = mels[bib, :, : mel_input_length[bib]].unsqueeze(0).unsqueeze(1)
            ss.append(model.predictor_encoder(mel))
            gs.append(model.style_encoder(mel))
        s_dur = torch.stack(ss).squeeze()
        gs = torch.stack(gs).squeeze()
        s_trg = torch.cat([gs, s_dur], dim=-1).detach()

        bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        # Diffusion
        loss_diff, loss_sty = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        if epoch >= loss_params.diff_epoch:
            num_steps = np.random.randint(3, 5)
            if self.config.model_params.multispeaker:
                s_preds = self.sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                    embedding=bert_dur,
                    features=ref,
                    num_steps=num_steps,
                ).squeeze(1)
                loss_diff = model.diffusion(
                    s_trg.unsqueeze(1), embedding=bert_dur, features=ref
                ).mean()
            else:
                s_preds = self.sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                    embedding=bert_dur,
                    num_steps=num_steps,
                ).squeeze(1)
                loss_diff = model.diffusion(
                    s_trg.unsqueeze(1), embedding=bert_dur
                ).mean()
            loss_sty = F.l1_loss(s_preds, s_trg.detach())

        # Predictor
        d, p = model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

        # Prepare clips for decoder
        mel_len = min(
            int(mel_input_length.min().item() / 2 - 1), self.config.max_len // 2
        )
        en_clip, p_en_clip, gt_clip, wav_clip, st_clip = [], [], [], [], []
        for bib in range(len(mel_input_length)):
            m_len = int(mel_input_length[bib].item() / 2)
            start = np.random.randint(0, m_len - mel_len)
            en_clip.append(asr[bib, :, start : start + mel_len])
            p_en_clip.append(p[bib, :, start : start + mel_len])
            gt_clip.append(mels[bib, :, (start * 2) : ((start + mel_len) * 2)])
            y = waves[bib][(start * 2) * 300 : ((start + mel_len) * 2) * 300]
            wav_clip.append(torch.from_numpy(y).to(device))

            st_start = np.random.randint(0, m_len - mel_len)
            st_clip.append(mels[bib, :, (st_start * 2) : ((st_start + mel_len) * 2)])

        wav_clip = torch.stack(wav_clip).float().detach().unsqueeze(1)
        en_clip = torch.stack(en_clip)
        p_en_clip = torch.stack(p_en_clip)
        gt_clip = torch.stack(gt_clip).detach()

        # Discriminator Step
        self.optimizer.zero_grad(key="msd")
        self.optimizer.zero_grad(key="mpd")

        with torch.no_grad():
            f0_real, _, _ = model.pitch_extractor(gt_clip.unsqueeze(1))
            n_real = log_norm(gt_clip.unsqueeze(1)).squeeze(1)
            s_clip = model.style_encoder(gt_clip.unsqueeze(1))

        f0_fake, n_fake = model.predictor.f0_n_train(
            p_en_clip, model.predictor_encoder(gt_clip.unsqueeze(1))
        )
        y_rec = model.decoder(en_clip, f0_fake, n_fake, s_clip)

        d_loss = self.dl(wav_clip.detach(), y_rec.detach())
        d_loss.backward()
        self.optimizer.step("msd")
        self.optimizer.step("mpd")

        # Generator Step
        self.optimizer.zero_grad()
        loss_mel = self.stft_loss(y_rec, wav_clip)
        loss_gen_all = self.gl(wav_clip, y_rec)
        loss_lm = self.wl(wav_clip.detach().squeeze(), y_rec.squeeze()).mean()

        # Duration & CE losses
        loss_dur, loss_ce = self._compute_dur_ce_losses(d, d_gt, input_lengths, texts)
        loss_s2s = F.cross_entropy(
            s2s_pred.flatten(0, 1), texts.flatten()
        )  # Simplified
        loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
        loss_f0_rec = F.smooth_l1_loss(f0_real, f0_fake) / 10
        loss_norm_rec = F.smooth_l1_loss(n_real, n_fake)

        g_loss = (
            loss_params.lambda_mel * loss_mel
            + loss_params.lambda_F0 * loss_f0_rec
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

        g_loss.backward()
        for key in [
            "bert_encoder",
            "bert",
            "predictor",
            "predictor_encoder",
            "style_encoder",
            "decoder",
            "text_encoder",
            "text_aligner",
            "diffusion",
        ]:
            if key in self.optimizer.keys:
                self.optimizer.step(key)

        # SLM Adversarial
        if epoch >= loss_params.joint_epoch:
            # Prepare inputs for SLM Adversarial Loss
            slm_out = self.slmadv(
                self.iters,
                wav_clip.detach(),
                y_rec.detach(),
                waves,
                mel_input_length,
                _ref_texts,
                _ref_lengths,
                ref,
            )

            if slm_out is not None:
                d_slm_loss, g_slm_loss, _ = slm_out

                # Backpropagate SLM Losses
                if d_slm_loss != 0:
                    d_slm_loss.backward()

                g_slm_loss.backward()

                # Record losses
                loss_dict["d_slm"] = d_slm_loss.item()
                loss_dict["g_slm"] = g_slm_loss.item()

        return {
            "mel_loss": loss_mel.item(),
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "dur_loss": loss_dur.item(),
            "f0_loss": loss_f0_rec.item(),
            "d_slm": loss_dict.get("d_slm", 0.0),
            "g_slm": loss_dict.get("g_slm", 0.0),
        }

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
            loss_ce += F.binary_cross_entropy_with_logits(
                _pred.flatten(), _trg.flatten()
            )
        return loss_dur / texts.size(0), loss_ce / texts.size(0)

    def _log_to_tensorboard(self, loss_dict, prefix):
        for k, v in loss_dict.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, self.iters)

    def _validate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for _batch in self.val_dataloader:
                # Simplified validation
                pass
        # Log validation results
        pass

    def _save_checkpoint(self, epoch):
        save_path = osp.join(self.config.log_dir, f"checkpoint_epoch_{epoch}.pth")
        state = {
            "net": {
                k: v.state_dict()
                for k, v in self.model._modules.items()
                if v is not None
            },
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "config": self.config.to_dict(),
        }
        torch.save(state, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
