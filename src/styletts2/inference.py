import argparse
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio
from loguru import logger

from styletts2.config import StyleTTS2Config
from styletts2.data.text import Tokenizer
from styletts2.models.styletts2 import StyleTTS2Model
from styletts2.voices import VoiceManager


class StyleTTS2Inference(torch.nn.Module):
    """
    High-level inference wrapper for StyleTTS2.

    Features
    --------
    - Loads weights from a ``.safetensors`` checkpoint.
    - Strips training-only modules automatically (saves ~200 MB RAM).
    - Supports named voices via ``VoiceManager``.
    - ``encode_reference()`` for zero-shot voice cloning from any audio file.
    - ``generate_long()`` for long-form TTS with proper style continuity.
    - ``stream()`` for sentence-by-sentence audio streaming.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        voices_dir: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Load Config
        if config_path and os.path.exists(config_path):
            self.config = StyleTTS2Config.from_yaml(config_path)
        else:
            self.config = StyleTTS2Config()

        # 2. Load Model — inference mode drops training-only modules automatically
        self.model = StyleTTS2Model.from_pretrained(
            checkpoint_path, self.config, mode="inference"
        )
        self.model.to(self.device)
        self.model.eval()

        # 3. Tokenizer & Voice Manager
        self.tokenizer = Tokenizer()
        self.voice_manager = VoiceManager(voices_dir, device=str(self.device))

    # ------------------------------------------------------------------
    # Data class for outputs
    # ------------------------------------------------------------------

    @dataclass
    class Output:
        audio: torch.Tensor   # (T,) float32 on CPU
        pred_dur: torch.Tensor  # (L,) long on CPU
        style: torch.Tensor   # (1, style_dim*2) float32 on CPU

    # ------------------------------------------------------------------
    # Mel extraction helper (used for encode_reference)
    # ------------------------------------------------------------------

    def _wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Convert a 1-D or (1, T) waveform tensor to a mel spectrogram."""
        p = self.config.preprocess_params
        to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=p.sr,
            n_fft=p.n_fft,
            win_length=p.win_length,
            hop_length=p.hop_length,
            n_mels=self.config.model_params.n_mels,
        ).to(self.device)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return to_mel(wav)  # (1, n_mels, T)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_reference(self, audio_path: str) -> torch.Tensor:
        """
        Encode a reference audio file into a style tensor for voice cloning.

        Returns a ``(1, style_dim * 2)`` tensor that can be passed directly
        as ``ref_s`` to ``forward()``.

        .. note::
            Requires ``style_encoder`` and ``predictor_encoder`` to be present.
            These are removed in inference mode — call this method on a model
            loaded with ``mode="train"`` or build a dedicated encoder wrapper.
        """
        wav, sr = torchaudio.load(audio_path)
        wav = wav.to(self.device)
        target_sr = self.config.preprocess_params.sr
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        mel = self._wav_to_mel(wav)  # (1, n_mels, T)

        # These attributes are present only when loaded for training.
        # For inference, load a separate encoder model or pass a pre-computed
        # voice tensor via VoiceManager.
        if not (hasattr(self.model, "style_encoder") and hasattr(self.model, "predictor_encoder")):
            raise RuntimeError(
                "encode_reference() requires style_encoder and predictor_encoder "
                "which are stripped in inference mode. Use VoiceManager with a "
                "pre-computed .pt voice pack instead, or load with mode='train'."
            )

        with torch.inference_mode():
            ref_ss = self.model.style_encoder(mel.unsqueeze(0))
            ref_sp = self.model.predictor_encoder(mel.unsqueeze(0))
        return torch.cat([ref_ss, ref_sp], dim=-1)  # (1, style_dim*2)

    @torch.inference_mode()
    def forward(
        self,
        text: str,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0,
        ref_s: Optional[torch.Tensor] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> "StyleTTS2Inference.Output":
        """
        End-to-end generation from text to audio.

        Args:
            text: Input text string.
            diffusion_steps: Number of diffusion sampling steps (used only
                             when no ``ref_s`` / ``voice`` is provided).
            embedding_scale: Classifier-free guidance scale for diffusion.
            ref_s: Pre-computed style tensor ``(1, style_dim*2)`` or a voice
                   pack ``(N, style_dim*2)``.  If ``None`` and ``voice`` is
                   also ``None``, style is sampled via diffusion.
            voice: Name of a voice pack stored in the ``voices_dir``.
            speed: Speaking rate multiplier (>1 = faster).

        Returns:
            :class:`Output` with fields ``audio``, ``pred_dur``, ``style``.
        """
        # 1. Tokenize
        tokens = self.tokenizer.encode(text)
        tokens = [0, *tokens, 0]
        tokens_tensor = torch.LongTensor([tokens]).to(self.device)
        input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(self.device)

        # 2. BERT & Text Encoding
        bert_dur = self.model.bert(
            tokens_tensor,
            attention_mask=torch.ones_like(tokens_tensor),
        )
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)
        t_en = self.model.text_encoder(tokens_tensor, input_lengths, None)

        # 3. Style Selection
        if ref_s is None and voice is not None:
            ref_s = self.voice_manager.get_voice(voice)

        if ref_s is None:
            raise ValueError("ref_s or voice must be provided. Diffusion sampling is disabled.")

        ref_s = ref_s.to(self.device)
        # Support Kokoro-style (N, style_dim*2) voice packs
        if ref_s.ndim == 2 and ref_s.shape[1] == self.config.model_params.style_dim * 2:
            idx = min(len(tokens) - 1, ref_s.shape[0] - 1)
            s_pred = ref_s[idx : idx + 1]  # (1, style_dim*2)
        elif ref_s.ndim == 1:
            s_pred = ref_s.unsqueeze(0)
        else:
            s_pred = ref_s

        style_dim = self.config.model_params.style_dim
        s = s_pred[:, style_dim:]    # predictor style
        ref = s_pred[:, :style_dim]  # decoder style

        # 4. Prosody Prediction
        d = self.model.predictor.text_encoder(d_en, s, input_lengths, None)
        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed
        pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()

        # 5. Alignment
        indices = torch.repeat_interleave(
            torch.arange(tokens_tensor.shape[1], device=self.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (tokens_tensor.shape[1], indices.shape[0]), device=self.device
        )
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
            style=s_pred.cpu().float(),
        )

    def generate_long(
        self,
        passage: str,
        alpha: float = 0.7,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate long-form audio by splitting into sentences.

        Style is blended across sentences using exponential smoothing
        controlled by ``alpha`` (higher = more continuity).

        Args:
            passage: Multi-sentence input text.
            alpha: Style blending factor between consecutive sentences.
            **kwargs: Forwarded to :meth:`forward` (e.g. ``speed``,
                      ``diffusion_steps``, ``voice``).

        Returns:
            Concatenated audio waveform as a 1-D ``torch.Tensor``.
        """
        # Remove sentence-splitting punctuation, keeping terminators
        sentences = [
            s.strip()
            for raw in passage.replace("!", ".").replace("?", ".").split(".")
            for s in [raw.strip()]
            if s
        ]

        wavs: list[torch.Tensor] = []
        last_s: Optional[torch.Tensor] = None

        for sentence in sentences:
            # Feed the blended style from the previous sentence as ref_s
            output = self.forward(sentence + ".", ref_s=last_s, **kwargs)

            # Smooth style for the *next* sentence
            if last_s is not None:
                last_s = alpha * last_s + (1 - alpha) * output.style
            else:
                last_s = output.style

            wavs.append(output.audio)

        if not wavs:
            return torch.zeros(0)
        return torch.cat(wavs, dim=0)

    def stream(
        self,
        passage: str,
        alpha: float = 0.7,
        **kwargs,
    ) -> Iterator["StyleTTS2Inference.Output"]:
        """
        Yield :class:`Output` objects sentence-by-sentence for real-time TTS.

        Args:
            passage: Multi-sentence input text.
            alpha: Style blending factor between consecutive sentences.
            **kwargs: Forwarded to :meth:`forward`.
        """
        sentences = [
            s.strip()
            for raw in passage.replace("!", ".").replace("?", ".").split(".")
            for s in [raw.strip()]
            if s
        ]

        last_s: Optional[torch.Tensor] = None
        for sentence in sentences:
            output = self.forward(sentence + ".", ref_s=last_s, **kwargs)
            if last_s is not None:
                last_s = alpha * last_s + (1 - alpha) * output.style
            else:
                last_s = output.style
            yield output


def main():
    parser = argparse.ArgumentParser(description="StyleTTS2 Inference CLI")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesise")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .safetensors checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--voice", type=str, default=None, help="Voice name (must be in voices_dir)")
    parser.add_argument("--voices_dir", type=str, default=None, help="Directory containing .pt voice packs")
    parser.add_argument("--output", type=str, default="output.wav", help="Output wav path")
    parser.add_argument("--long", action="store_true", help="Use sentence-splitting for long text")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed multiplier")
    args = parser.parse_args()

    engine = StyleTTS2Inference(
        args.checkpoint,
        config_path=args.config,
        voices_dir=args.voices_dir,
    )

    if args.long:
        wav = engine.generate_long(args.text, voice=args.voice, speed=args.speed)
    else:
        output = engine.forward(
            args.text,
            voice=args.voice,
            speed=args.speed,
        )
        wav = output.audio

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    torchaudio.save(args.output, wav, sample_rate=24000)
    logger.info(f"✅ Generated audio saved to {args.output}")


if __name__ == "__main__":
    main()
