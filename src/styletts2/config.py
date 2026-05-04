from dataclasses import dataclass, field
from typing import Any, Optional

import yaml
from transformers import PretrainedConfig


@dataclass
class PreprocessConfig:
    sr: int = 24000
    n_fft: int = 2048
    win_length: int = 1200
    hop_length: int = 300


@dataclass
class DecoderConfig:
    type: str = "hifigan"
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    upsample_rates: list[int] = field(default_factory=lambda: [10, 5, 3, 2])
    upsample_initial_channel: int = 512
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [20, 10, 6, 4])
    gen_istft_n_fft: int = 20
    gen_istft_hop_size: int = 5


@dataclass
class SLMConfig:
    model: str = "microsoft/wavlm-base-plus"
    sr: int = 16000
    hidden: int = 768
    nlayers: int = 13
    initial_channel: int = 64


@dataclass
class TransformerConfig:
    num_layers: int = 3
    num_heads: int = 8
    head_features: int = 64
    multiplier: int = 2


@dataclass
class DiffusionDistConfig:
    sigma_data: float = 0.2
    estimate_sigma_data: bool = True
    mean: float = -3.0
    std: float = 1.0


@dataclass
class DiffusionConfig:
    embedding_mask_proba: float = 0.1
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    dist: DiffusionDistConfig = field(default_factory=DiffusionDistConfig)


@dataclass
class ModelParamsConfig:
    multispeaker: bool = True
    dim_in: int = 64
    hidden_dim: int = 512
    max_conv_dim: int = 512
    n_layer: int = 3
    n_mels: int = 80
    n_token: int = 178
    max_dur: int = 50
    style_dim: int = 128
    dropout: float = 0.2
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    slm: SLMConfig = field(default_factory=SLMConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


@dataclass
class LossParamsConfig:
    lambda_mel: float = 5.0
    lambda_gen: float = 1.0
    lambda_slm: float = 1.0
    lambda_mono: float = 1.0
    lambda_s2s: float = 1.0
    lambda_f0: float = 1.0
    lambda_norm: float = 1.0
    lambda_dur: float = 1.0
    lambda_ce: float = 20.0
    lambda_sty: float = 1.0
    lambda_diff: float = 1.0
    diff_epoch: int = 10
    joint_epoch: int = 30


@dataclass
class OptimizerParamsConfig:
    lr: float = 0.0001
    bert_lr: float = 0.00001
    ft_lr: float = 0.0001


@dataclass
class SLMAdvParamsConfig:
    min_len: int = 400
    max_len: int = 500
    batch_percentage: float = 0.5
    iter: int = 10
    thresh: float = 5.0
    scale: float = 0.01
    sig: float = 1.5


@dataclass
class DataParamsConfig:
    train_data: str = "Data/train_list.txt"
    val_data: str = "Data/val_list.txt"
    root_path: str = ""
    ood_data: str = "Data/OOD_texts.txt"
    min_length: int = 50


@dataclass
class ASRConfig:
    input_dim: int = 80
    hidden_dim: int = 256
    n_token: int = 178
    token_embedding_dim: int = 512


@dataclass
class PLBERTConfig:
    vocab_size: int = 178
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    max_position_embeddings: int = 512
    num_hidden_layers: int = 12
    dropout: float = 0.1


@dataclass
class ExternalModelsConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    plbert: PLBERTConfig = field(default_factory=PLBERTConfig)
    # JDC is simple enough to not need a separate config class here for now


class StyleTTS2Config(PretrainedConfig):
    model_type = "styletts2"

    def __init__(
        self,
        log_dir: str = "Models/Output",
        save_freq: int = 5,
        log_interval: int = 10,
        device: str = "cuda",
        epochs: int = 50,
        batch_size: int = 8,
        max_len: int = 400,
        pretrained_model: str = "",
        second_stage_load_pretrained: bool = True,
        load_only_params: bool = True,
        external_models: Optional[dict[str, Any]] = None,
        data_params: Optional[dict[str, Any]] = None,
        preprocess_params: Optional[dict[str, Any]] = None,
        model_params: Optional[dict[str, Any]] = None,
        loss_params: Optional[dict[str, Any]] = None,
        optimizer_params: Optional[dict[str, Any]] = None,
        slmadv_params: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.log_interval = log_interval
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.pretrained_model = pretrained_model
        self.second_stage_load_pretrained = second_stage_load_pretrained
        self.load_only_params = load_only_params

        ext_data = external_models or {}
        asr_data = ext_data.pop("asr", {})
        plbert_data = ext_data.pop("plbert", {})
        self.external_models = ExternalModelsConfig(
            asr=ASRConfig(**asr_data), plbert=PLBERTConfig(**plbert_data)
        )

        self.data_params = DataParamsConfig(**(data_params or {}))

        # Preprocess params
        preprocess_data = preprocess_params or {}
        spect_params = preprocess_data.pop("spect_params", {})
        self.preprocess_params = PreprocessConfig(**preprocess_data, **spect_params)

        # Model params
        model_data = model_params or {}
        decoder_data = model_data.pop("decoder", {})
        slm_data = model_data.pop("slm", {})
        diffusion_data = model_data.pop("diffusion", {})

        diff_transformer_data = diffusion_data.pop("transformer", {})
        diff_dist_data = diffusion_data.pop("dist", {})

        diffusion_config = DiffusionConfig(
            embedding_mask_proba=diffusion_data.get("embedding_mask_proba", 0.1),
            transformer=TransformerConfig(**diff_transformer_data),
            dist=DiffusionDistConfig(**diff_dist_data),
        )

        self.model_params = ModelParamsConfig(
            **model_data,
            decoder=DecoderConfig(**decoder_data),
            slm=SLMConfig(**slm_data),
            diffusion=diffusion_config,
        )

        self.loss_params = LossParamsConfig(**(loss_params or {}))
        self.optimizer_params = OptimizerParamsConfig(**(optimizer_params or {}))
        self.slmadv_params = SLMAdvParamsConfig(**(slmadv_params or {}))

    @classmethod
    def from_yaml(cls, yaml_path: str):

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
