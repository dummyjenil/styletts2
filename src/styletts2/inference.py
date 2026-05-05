import argparse
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from loguru import logger

from styletts2.config import StyleTTS2Config
from styletts2.data.text import Tokenizer
from styletts2.models.diffusion.k_diffusion import ADPM2Sampler, DiffusionSampler
from styletts2.models.diffusion.sampler import KarrasSchedule
from styletts2.models.styletts2 import StyleTTS2Model


class StyleTTS2Inference(torch.nn.Module):
    """
    StyleTTS2Inference is a high-level wrapper for inference, inspired by the KModel architecture.
    It handles:
    1. Loading weights and config (optionally from HF).
    2. Text tokenization.
    3. End-to-end generation from phonemes/text to audio.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Load Config
        if config_path and os.path.exists(config_path):
            self.config = StyleTTS2Config.from_yaml(config_path)
        # Fallback to default or download if repo_id is provided
        elif repo_id:
            logger.info(f"Downloading config from HF repo: {repo_id}")
            cfg_file = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(cfg_file):
                # In a real scenario, we might need to map Kokoro-style JSON to StyleTTS2Config
                # For now, we assume StyleTTS2Config can handle it or use defaults
                self.config = StyleTTS2Config()
        else:
            self.config = StyleTTS2Config()

        # 2. Load Model
        self.model = StyleTTS2Model.from_pretrained(checkpoint_path, self.config)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = Tokenizer()

        # 3. Initialize Sampler (Diffusion)
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

    @dataclass
    class Output:
        audio: torch.Tensor
        pred_dur: torch.Tensor
        style: torch.Tensor

    @torch.no_grad()
    def forward(
        self,
        text: str,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0,
        ref_s: Optional[torch.FloatTensor] = None,
        alpha: float = 0.7,
        speed: float = 1.0,
    ) -> Output:
        """
        End-to-end generation.
        """
        # 1. Tokenize
        tokens = self.tokenizer.encode(text)
        # Add [CLS] and [SEP] tokens (0 in our current tokenizer)
        tokens = [0, *tokens, 0]
        tokens_tensor = torch.LongTensor([tokens]).to(self.device)
        input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(self.device)

        # 2. BERT & Text Encoding
        bert_dur = self.model.bert(tokens_tensor, attention_mask=torch.ones_like(tokens_tensor))
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)
        t_en = self.model.text_encoder(tokens_tensor, input_lengths, None) # Mask is handled internally if None

        # 3. Style Generation (Diffusion)
        if ref_s is None:
            noise = torch.randn(1, 1, self.config.model_params.style_dim * 2).to(self.device)
            s_pred = self.sampler(
                noise,
                num_steps=diffusion_steps,
                embedding=bert_dur,
                embedding_scale=embedding_scale,
            ).squeeze(1)
        else:
            s_pred = ref_s

        s = s_pred[:, self.config.model_params.style_dim :]
        ref = s_pred[:, : self.config.model_params.style_dim]

        # 4. Prosody Prediction
        d = self.model.predictor.text_encoder(d_en, s, input_lengths, None)
        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed
        pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()

        # 5. Alignment
        indices = torch.repeat_interleave(torch.arange(tokens_tensor.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((tokens_tensor.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        # 6. Prosody & Decoding
        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = self.model.predictor.f0_n_train(en, s)

        asr = t_en @ pred_aln_trg
        audio = self.model.decoder(asr, f0_pred, n_pred, ref)

        return self.Output(
            audio=audio.squeeze().cpu().float(),
            pred_dur=pred_dur.cpu().long(),
            style=s_pred.cpu().float()
        )

    def generate_long(self, passage: str, **kwargs) -> torch.Tensor:
        """
        Generate long-form audio by splitting into sentences.
        """
        sentences = passage.replace("!", ".").replace("?", ".").split(".")
        wavs = []
        last_s = None
        alpha = kwargs.get("alpha", 0.7)

        for sentence in sentences:
            if not sentence.strip():
                continue

            output = self.forward(sentence.strip() + ".", **kwargs)

            if last_s is not None:
                # Smooth style transitions
                output.style = alpha * last_s + (1 - alpha) * output.style

            wavs.append(output.audio)
            last_s = output.style

        return torch.cat(wavs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()

    engine = StyleTTS2Inference(args.checkpoint, args.config)

    if args.long:
        wav = engine.generate_long(args.text)
    else:
        output = engine.forward(args.text)
        wav = output.audio

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    torchaudio.save(args.output, wav, sample_rate=24000)
    print(f"✅ Generated audio saved to {args.output}")


if __name__ == "__main__":
    main()

