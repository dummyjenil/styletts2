from typing import Self

import torch
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

        # --- Decoder Selection ---
        if params.decoder.type == "istftnet":
            decoder_params = params.decoder.__dict__.copy()
            decoder_params.pop("type")
            self.decoder = ISTFTDecoder(
                dim_in=params.hidden_dim,
                style_dim=params.style_dim,
                dim_out=params.n_mels,
                **decoder_params,
            )
        else:
            self.decoder = HiFiGANDecoder(
                dim_in=params.hidden_dim,
                style_dim=params.style_dim,
                dim_out=params.n_mels,
                resblock_kernel_sizes=params.decoder.resblock_kernel_sizes,
                upsample_rates=params.decoder.upsample_rates,
                upsample_initial_channel=params.decoder.upsample_initial_channel,
                resblock_dilation_sizes=params.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=params.decoder.upsample_kernel_sizes,
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

    def forward(self, *args, **kwargs):
        """Forward pass for training or inference can be implemented here"""
        pass

    @classmethod
    def from_pretrained(cls, path: str, config: StyleTTS2Config) -> Self:
        model = cls(config)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model
