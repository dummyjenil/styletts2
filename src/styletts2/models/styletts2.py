from typing import Literal, Optional, Self

from safetensors.torch import load_file as safetensors_load_file
from torch import nn
from transformers import AlbertConfig

from styletts2.config import StyleTTS2Config
from styletts2.external import ASRCNN, CustomAlbert, JDCNet
from styletts2.models.decoders.hifigan import HiFiGANDecoder
from styletts2.models.decoders.istftnet import ISTFTDecoder
from styletts2.models.diffusion.k_diffusion import KDiffusion
from styletts2.models.diffusion.model import StyleTransformer1d, Transformer1d
from styletts2.models.diffusion.sampler import (
    AudioDiffusionConditional,
    LogNormalDistribution,
)
from styletts2.models.discriminators.mpd import MultiPeriodDiscriminator
from styletts2.models.discriminators.msd import MultiResSpecDiscriminator
from styletts2.models.discriminators.wavlm import WavLMDiscriminator
from styletts2.models.encoders.prosody import ProsodyPredictor
from styletts2.models.encoders.style import StyleEncoder
from styletts2.models.encoders.text import TextEncoder

# Components only needed during training
_TRAINING_ONLY_ATTRS = [
    "wd",
    "msd",
    "mpd",
    "pitch_extractor",
    "text_aligner",
    "diffusion",
    "predictor_encoder",
    "style_encoder",
]


class StyleTTS2Model(nn.Module):
    def __init__(self, config: StyleTTS2Config):
        super().__init__()
        self.config = config
        params = config.model_params

        # --- Core Text & BERT Modules ---
        bert_config = AlbertConfig(**config.external_models.plbert.__dict__)
        self.bert = CustomAlbert(bert_config)
        self.bert_encoder = nn.Linear(
            config.external_models.plbert.hidden_size, params.hidden_dim
        )

        self.text_encoder = TextEncoder(
            channels=params.hidden_dim,
            kernel_size=5,
            depth=params.n_layer,
            n_symbols=params.n_token,
        )

        # --- Prosody & Style Modules ---
        self.predictor = ProsodyPredictor(
            style_dim=params.style_dim,
            d_hid=params.hidden_dim,
            nlayers=params.n_layer,
            max_dur=params.max_dur,
            dropout=params.dropout,
        )

        # --- Decoder Selection ---
        if params.decoder.type == "istftnet":
            decoder_params = params.decoder.__dict__.copy()
            decoder_params.pop("type")
            self.decoder = ISTFTDecoder(
                dim_in=params.hidden_dim,
                style_dim=params.style_dim,
                **decoder_params,
            )
        else:
            self.decoder = HiFiGANDecoder(
                dim_in=params.hidden_dim,
                style_dim=params.style_dim,
                resblock_kernel_sizes=params.decoder.resblock_kernel_sizes,
                upsample_rates=params.decoder.upsample_rates,
                upsample_initial_channel=params.decoder.upsample_initial_channel,
                resblock_dilation_sizes=params.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=params.decoder.upsample_kernel_sizes,
            )

        self.style_encoder = StyleEncoder(
            dim_in=params.dim_in,
            style_dim=params.style_dim,
            max_conv_dim=params.hidden_dim,
        )

        self.predictor_encoder = StyleEncoder(
            dim_in=params.dim_in,
            style_dim=params.style_dim,
            max_conv_dim=params.hidden_dim,
        )

        # --- Diffusion Style Predictor ---
        diff_params = params.diffusion
        if params.multispeaker:
            transformer = StyleTransformer1d(
                channels=params.style_dim * 2,
                context_embedding_features=config.external_models.plbert.hidden_size,
                context_features=params.style_dim * 2,
                num_layers=diff_params.transformer.num_layers,
                num_heads=diff_params.transformer.num_heads,
                head_features=diff_params.transformer.head_features,
                multiplier=diff_params.transformer.multiplier,
            )
        else:
            transformer = Transformer1d(
                channels=params.style_dim * 2,
                context_embedding_features=config.external_models.plbert.hidden_size,
                num_layers=diff_params.transformer.num_layers,
                num_heads=diff_params.transformer.num_heads,
                head_features=diff_params.transformer.head_features,
                multiplier=diff_params.transformer.multiplier,
            )

        self.diffusion = AudioDiffusionConditional(
            in_channels=1,
            embedding_max_length=config.external_models.plbert.max_position_embeddings,
            embedding_features=config.external_models.plbert.hidden_size,
            embedding_mask_proba=diff_params.embedding_mask_proba,
            channels=params.style_dim * 2,
            context_features=params.style_dim * 2,
        )

        self.diffusion.diffusion = KDiffusion(
            net=transformer,
            sigma_distribution=LogNormalDistribution(
                mean=diff_params.dist.mean, std=diff_params.dist.std
            ),
            sigma_data=diff_params.dist.sigma_data,
            dynamic_threshold=0.0,
        )
        self.diffusion.unet = transformer

        # --- External Aligner & Pitch Extractor ---
        self.text_aligner = ASRCNN(**config.external_models.asr.__dict__)
        self.pitch_extractor = JDCNet(num_class=1, seq_len=192)

        # --- Discriminators ---
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiResSpecDiscriminator()
        self.wd = WavLMDiscriminator(
            params.slm.hidden, params.slm.nlayers, params.slm.initial_channel
        )

    def _drop_training_components(self) -> None:
        """Remove training-only modules to save RAM during inference."""
        for attr in _TRAINING_ONLY_ATTRS:
            if hasattr(self, attr):
                delattr(self, attr)

    def forward(self, *args, **kwargs):
        """Forward pass for training or inference can be implemented here"""
        pass

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        path: Optional[str],
        config: StyleTTS2Config,
        mode: Literal["inference", "train"] = "inference",
    ) -> Self:
        """
        Load a model from a safetensors checkpoint.

        Args:
            path: Path to the .safetensors checkpoint file.
            config: StyleTTS2Config instance.
            mode: ``"inference"`` strips training-only modules (wd, msd, mpd,
                  pitch_extractor, text_aligner, diffusion, predictor_encoder,
                  style_encoder) after loading to save RAM.
                  ``"train"`` keeps all modules.
        """
        model = cls(config)
        if mode == "inference":
            model._drop_training_components()
        if path:
            state = safetensors_load_file(path, device="cpu")
            model.load_state_dict(state)
        return model

    @classmethod
    def from_pretrained_split(
        cls,
        slim_path: str,
        large_path: str,
        config: StyleTTS2Config,
    ) -> Self:
        """
        Load the generator weights from a slim safetensors checkpoint and the
        training-only components (text_aligner, pitch_extractor, wd, msd, mpd)
        from a second large safetensors checkpoint.

        Both paths must point to ``.safetensors`` files.
        """
        model = cls(config)

        # Generator from slim model
        slim_state = safetensors_load_file(slim_path, device="cpu")
        model.load_state_dict(slim_state, strict=False)

        # Training components from large model
        large_state = safetensors_load_file(large_path, device="cpu")
        training_keys = ["text_aligner.", "pitch_extractor.", "wd.", "msd.", "mpd."]
        filtered = {
            k: v
            for k, v in large_state.items()
            if any(k.startswith(prefix) for prefix in training_keys)
        }
        model.load_state_dict(filtered, strict=False)
        return model

    @classmethod
    # ruff: noqa: PLR0912
    def load_finetune_mode(
        cls,
        slim_path: str,
        large_path: str,
        config: StyleTTS2Config,
        mode: Literal[
            "full", "decoder_only", "speaker_only", "diffusion_only", "no_discriminators"
        ] = "full",
    ) -> Self:
        """
        Convenience factory for fine-tuning.  Loads weights via
        ``from_pretrained_split`` then freezes the appropriate sub-modules
        according to ``mode``.

        Modes
        -----
        full              - Everything trainable (default).
        decoder_only      - Freeze bert, text_encoder, predictor, diffusion.
        speaker_only      - Freeze everything except style_encoder and
                            predictor_encoder.
        diffusion_only    - Freeze generator; only train the diffusion
                            style predictor.
        no_discriminators - Full generator but discriminators are NOT trained
                            (safe for small datasets).
        """
        model = cls.from_pretrained_split(slim_path, large_path, config)

        if mode == "full":
            pass  # All trainable

        elif mode == "decoder_only":
            for module in [
                model.bert,
                model.bert_encoder,
                model.text_encoder,
                model.predictor,
                model.diffusion,
            ]:
                for p in module.parameters():
                    p.requires_grad_(False)

        elif mode == "speaker_only":
            for name, module in model.named_children():
                if name not in ("style_encoder", "predictor_encoder"):
                    for p in module.parameters():
                        p.requires_grad_(False)

        elif mode == "diffusion_only":
            for name, module in model.named_children():
                if name != "diffusion":
                    for p in module.parameters():
                        p.requires_grad_(False)

        elif mode == "no_discriminators":
            for module in [model.mpd, model.msd, model.wd]:
                for p in module.parameters():
                    p.requires_grad_(False)

        return model
