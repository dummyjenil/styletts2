import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from styletts2.utils.helpers import length_to_mask


class SLMAdversarialLoss(nn.Module):
    def __init__(
        self,
        model,
        wl,
        sampler,
        min_len=400,
        max_len=500,
        batch_percentage=0.5,
        skip_update=1,
        sig=1.5,
    ):
        super().__init__()
        self.model = model
        self.wl = wl
        self.sampler = sampler
        self.min_len = min_len
        self.max_len = max_len
        self.batch_percentage = batch_percentage
        self.skip_update = skip_update
        self.sig = sig

    def forward(
        self,
        iters,
        y_rec_gt,
        y_rec_gt_pred,
        waves,
        mel_input_length,
        ref_text,
        ref_lengths,
        ref_s,
    ):
        device = ref_text.device
        text_mask = length_to_mask(ref_lengths).to(device)

        # 1. Text Encoding & Duration Prediction
        t_en = self.model.text_encoder(ref_text, ref_lengths, text_mask)
        s_dur = ref_s[:, 128:]

        # Predict duration
        d, _ = self.model.predictor(
            t_en,
            s_dur,
            ref_lengths,
            torch.randn(ref_lengths.shape[0], ref_lengths.max(), 2, device=device),
            text_mask,
        )

        # 2. Differentiable Duration Modeling (Gaussian Alignment)
        output_lengths = []
        attn_preds = []

        for _s2s_pred, _text_length in zip(d, ref_lengths):
            _s2s_pred_org = _s2s_pred[:_text_length, :]
            _s2s_pred_sig = torch.sigmoid(_s2s_pred_org)
            _dur_pred = _s2s_pred_sig.sum(dim=-1)

            output_len = int(torch.round(_s2s_pred_sig.sum()).item())
            t = (
                torch.arange(0, output_len, device=device)
                .unsqueeze(0)
                .expand(len(_s2s_pred_sig), output_len)
            )
            loc = torch.cumsum(_dur_pred, dim=0) - _dur_pred / 2

            # Gaussian kernel
            h = torch.exp(
                -0.5 * torch.square(t - (output_len - loc.unsqueeze(-1))) / (self.sig**2)
            )

            # Alignment via convolution
            out = F.conv1d(
                _s2s_pred_org.unsqueeze(0),
                h.unsqueeze(1),
                padding=h.shape[-1] - 1,
                groups=int(_text_length),
            )[..., :output_len]

            attn_preds.append(F.softmax(out.squeeze(0), dim=0))
            output_lengths.append(output_len)

        if not output_lengths:
            return None

        max_out_len = max(output_lengths)

        # 3. Feature Generation
        with torch.no_grad():
            t_en_fixed = self.model.text_encoder(ref_text, ref_lengths, text_mask)

        s2s_attn = torch.zeros(
            len(ref_lengths), int(ref_lengths.max()), max_out_len, device=device
        )
        for i, (attn, length) in enumerate(zip(attn_preds, output_lengths)):
            s2s_attn[i, : ref_lengths[i], :length] = attn

        asr_pred = t_en_fixed @ s2s_attn

        # Prosody prediction
        _, p_pred = self.model.predictor(
            t_en_fixed, s_dur, ref_lengths, s2s_attn, text_mask
        )

        # 4. Clipping & Batch Preparation
        mel_len = max(int(min(output_lengths) / 2 - 1), self.min_len // 2)
        mel_len = min(mel_len, self.max_len // 2)

        en_clips, p_en_clips, sp_clips, wav_clips = [], [], [], []

        for i in range(len(output_lengths)):
            pred_len = output_lengths[i]
            gt_len = int(mel_input_length[i].item() / 2)

            if gt_len <= mel_len or pred_len <= mel_len:
                continue

            sp_clips.append(ref_s[i])

            # Random clips from prediction
            start_pred = np.random.randint(0, pred_len - mel_len)
            en_clips.append(asr_pred[i, :, start_pred : start_pred + mel_len])
            p_en_clips.append(p_pred[i, :, start_pred : start_pred + mel_len])

            # Random clips from ground truth
            start_gt = np.random.randint(0, gt_len - mel_len)
            y = waves[i][(start_gt * 2) * 300 : ((start_gt + mel_len) * 2) * 300]
            wav_clips.append(torch.from_numpy(y).to(device))

            if len(wav_clips) >= self.batch_percentage * len(waves):
                break

        if len(sp_clips) <= 1:
            return None

        sp_clips = torch.stack(sp_clips)
        wav_clips = torch.stack(wav_clips).float().unsqueeze(1)
        en_clips = torch.stack(en_clips)
        p_en_clips = torch.stack(p_en_clips)

        # 5. Decode
        f0_fake, n_fake = self.model.predictor.f0_n_train(
            p_en_clips, sp_clips[:, 128:].unsqueeze(-1)  # Matching trainer's style
        )
        # Note: In trainer, model.predictor_encoder(gt_clip.unsqueeze(1)) was used for style.
        # Here we use sp_clips[:, 128:] which is the predictor style.

        # Correction: model.predictor.f0_n_train takes (p, style)
        # In trainer.py: f0_fake, n_fake = model.predictor.f0_n_train(p_en_clip, model.predictor_encoder(gt_clip.unsqueeze(1)))
        # Here sp_clips[:, 128:] is already the predictor style from predictor_encoder.
        f0_fake, n_fake = self.model.predictor.f0_n_train(p_en_clips, sp_clips[:, 128:])

        y_pred = self.model.decoder(en_clips, f0_fake, n_fake, sp_clips[:, :128])

        # 6. Adversarial Loss Calculation
        d_loss = torch.tensor(0.0, device=device)
        if (iters + 1) % self.skip_update == 0:
            use_rec = np.random.randint(0, 2) == 0
            wav_target = y_rec_gt_pred if use_rec else wav_clips

            crop_size = min(wav_target.size(-1), y_pred.size(-1))

            if use_rec:
                # Length invariant regularization
                if wav_target.size(-1) > y_pred.size(-1):
                    real_crop = wav_target[..., :crop_size]
                    out_crop = self.wl.discriminator_forward(real_crop.detach().squeeze())
                    out_full = self.wl.discriminator_forward(
                        wav_target.detach().squeeze()
                    )
                    loss_reg = F.l1_loss(out_crop, out_full[..., : out_crop.size(-1)])
                else:
                    pred_crop = y_pred[..., :crop_size]
                    out_crop = self.wl.discriminator_forward(pred_crop.detach().squeeze())
                    out_full = self.wl.discriminator_forward(y_pred.detach().squeeze())
                    loss_reg = F.l1_loss(out_crop, out_full[..., : out_crop.size(-1)])

                d_loss = self.wl.discriminator(
                    wav_target.detach().squeeze(), y_pred.detach().squeeze()
                ).mean()
                d_loss += loss_reg

                # Artifact regularization
                out_gt = self.wl.discriminator_forward(y_rec_gt.detach().squeeze())
                out_rec = self.wl.discriminator_forward(y_rec_gt_pred.detach().squeeze())
                d_loss += F.l1_loss(out_gt, out_rec)
            else:
                d_loss = self.wl.discriminator(
                    wav_target.detach().squeeze(), y_pred.detach().squeeze()
                ).mean()

        # 7. Generator Loss
        gen_loss = self.wl.generator(y_pred.squeeze()).mean()

        return d_loss, gen_loss, y_pred.detach().cpu().numpy()
