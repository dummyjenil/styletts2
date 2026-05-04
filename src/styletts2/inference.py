import argparse

import torch
import torchaudio

from styletts2.config import StyleTTS2Config
from styletts2.data.text import Tokenizer
from styletts2.models.diffusion.k_diffusion import ADPM2Sampler, DiffusionSampler
from styletts2.models.diffusion.sampler import KarrasSchedule
from styletts2.models.styletts2 import StyleTTS2Model
from styletts2.utils.helpers import length_to_mask


class StyleTTS2Inference:
    def __init__(self, model_path, config_path=None, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load Config
        if config_path:
            self.config = StyleTTS2Config.from_yaml(config_path)
        else:
            self.config = StyleTTS2Config()

        # Load Model
        self.model = StyleTTS2Model.from_pretrained(model_path, self.config)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = Tokenizer()

        # Initialize Sampler
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

    @torch.no_grad()
    def generate(
        self, text, diffusion_steps=5, embedding_scale=1.0, s_prev=None, alpha=0.7
    ):
        # 1. Tokenize
        tokens = self.tokenizer.encode(text)
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
        text_mask = length_to_mask(input_lengths).to(self.device)

        # 2. Encoders
        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        # 3. Style Prediction (Diffusion)
        noise = torch.randn(1, 1, self.config.model_params.style_dim * 2).to(
            self.device
        )
        s_pred = self.sampler(
            noise,
            num_steps=diffusion_steps,
            embedding=bert_dur,
            embedding_scale=embedding_scale,
        ).squeeze(1)

        if s_prev is not None:
            # Long-form style combination
            s_pred = alpha * s_prev + (1 - alpha) * s_pred

        s = s_pred[:, self.config.model_params.style_dim :]
        ref = s_pred[:, : self.config.model_params.style_dim]

        # 4. Prosody Prediction
        # Using submodules to match notebook logic
        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        # 5. Manual Alignment
        pred_aln_trg = torch.zeros(
            int(input_lengths.item()), int(pred_dur.sum().item())
        ).to(self.device)
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].item())] = 1
            c_frame += int(pred_dur[i].item())

        # 6. Prosody & Decoding
        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
        f0_pred, n_pred = self.model.predictor.f0_n_train(en, s)

        out = self.model.decoder(
            (t_en @ pred_aln_trg.unsqueeze(0)),
            f0_pred,
            n_pred,
            ref.squeeze().unsqueeze(0),
        )

        return out.squeeze().cpu(), s_pred

    def generate_long(
        self, passage, diffusion_steps=10, embedding_scale=1.5, alpha=0.7
    ):
        # Simple split by punctuation for demo
        sentences = passage.replace("!", ".").replace("?", ".").split(".")
        wavs = []
        s_prev = None
        for sentence in sentences:
            if sentence.strip() == "":
                continue
            sentence_text = sentence + "."
            wav, s_prev = self.generate(
                sentence_text, diffusion_steps, embedding_scale, s_prev, alpha
            )
            wavs.append(wav)
        return torch.cat(wavs, dim=0)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument(
        "--long", action="store_true", help="Treat input as long passage"
    )
    args = parser.parse_args()

    engine = StyleTTS2Inference(args.checkpoint, args.config)

    if args.long:
        wav = engine.generate_long(args.text)
    else:
        wav, _ = engine.generate(args.text)

    # torchaudio.save expects [channels, time]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(args.output, wav, sample_rate=24000)
    print(f"✅ Generated audio saved to {args.output}")


if __name__ == "__main__":
    main()
