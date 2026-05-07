import argparse
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from styletts2.config import StyleTTS2Config
from styletts2.data.lightning_data import StyleTTS2DataModule
from styletts2.models.styletts2 import StyleTTS2Model
from styletts2.trainer.lightning_module import StyleTTS2LightningModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FINETUNE_MODES = [
    "full",
    "decoder_only",
    "speaker_only",
    "diffusion_only",
    "no_discriminators",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="StyleTTS2 Fine-tuning with PyTorch Lightning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────
    g = parser.add_argument_group("Data")
    g.add_argument("--train_data", type=str, default=None,
                   help="Path / HF dataset id for training data")
    g.add_argument("--val_data", type=str, default=None,
                   help="Path / HF dataset id for validation data")
    g.add_argument("--batch_size", type=int, default=4)
    g.add_argument("--num_workers", type=int, default=4)

    # ── Model ─────────────────────────────────────────────────────────
    g = parser.add_argument_group("Model")
    g.add_argument("--config", type=str, default=None,
                   help="Path to config.yaml (uses defaults when omitted)")
    g.add_argument("--pretrained_model", type=str, default=None,
                   help="Path to slim .safetensors generator checkpoint")
    g.add_argument("--large_model", type=str, default=None,
                   help="Path to large .safetensors checkpoint to borrow "
                        "text_aligner / pitch_extractor / discriminators from")
    g.add_argument("--mode", type=str, default="full", choices=FINETUNE_MODES,
                   help=(
                       "Finetune mode:\n"
                       "  full             - train everything\n"
                       "  decoder_only     - freeze bert/text_enc/predictor/diffusion\n"
                       "  speaker_only     - only train style_encoder + predictor_encoder\n"
                       "  diffusion_only   - only train the diffusion style predictor\n"
                       "  no_discriminators - skip discriminator training (safe for small datasets)"
                   ))

    # ── Training ──────────────────────────────────────────────────────
    g = parser.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--devices", type=int, default=1,
                   help="Number of GPUs (0 = CPU)")
    g.add_argument("--precision", type=str, default="32",
                   choices=["32", "16-mixed", "bf16-mixed"],
                   help="Training precision")
    g.add_argument("--grad_clip", type=float, default=10.0,
                   help="Gradient clipping max norm (0 = disabled)")
    g.add_argument("--resume", type=str, default=None,
                   help="Path to a Lightning .ckpt file to resume from")

    # ── Logging ───────────────────────────────────────────────────────
    g = parser.add_argument_group("Logging")
    g.add_argument("--log_dir", type=str, default="logs")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        config = StyleTTS2Config.from_yaml(args.config)
    else:
        logger.info("No --config provided, using default StyleTTS2Config.")
        config = StyleTTS2Config()

    # Override with CLI values
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.log_dir = args.log_dir

    # ── Data Module ───────────────────────────────────────────────────
    data_module = StyleTTS2DataModule(
        config=config,
        train_data=args.train_data or config.data_params.train_data,
        val_data=args.val_data or config.data_params.val_data or None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────
    slim_path = args.pretrained_model or config.pretrained_model or None
    large_path = args.large_model

    if slim_path and large_path:
        logger.info(
            f"Loading generator from {slim_path}, "
            f"training components from {large_path} (mode={args.mode})"
        )
        base_model = StyleTTS2Model.load_finetune_mode(
            slim_path, large_path, config, mode=args.mode
        )
    elif slim_path:
        logger.info(f"Loading pretrained model from {slim_path} (train mode, mode={args.mode})")
        base_model = StyleTTS2Model.from_pretrained(slim_path, config, mode="train")
        _apply_freeze(base_model, args.mode)
    else:
        logger.info("No pretrained model — training from scratch.")
        base_model = StyleTTS2Model(config)

    # ── Lightning Module ──────────────────────────────────────────────
    model = StyleTTS2LightningModule(config=config, model=base_model)

    # ── Callbacks ────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "checkpoints"),
        filename="styletts2-{epoch:02d}-{val/mel_loss:.4f}",
        save_top_k=3,
        monitor="val/mel_loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ── Logger ────────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name="tensorboard")

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=args.devices,
        precision=args.precision,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=config.log_interval,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        gradient_clip_algorithm="norm",
    )

    # ── Train ─────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)


# ruff: noqa: PLR0912
def _apply_freeze(model: StyleTTS2Model, mode: str) -> None:
    """Apply parameter freezing when loading from a single slim checkpoint."""

    if mode == "full":
        return
    if mode == "decoder_only":
        for m in [model.bert, model.bert_encoder, model.text_encoder,
                  model.predictor, model.diffusion]:
            for p in m.parameters():
                p.requires_grad_(False)
    elif mode == "speaker_only":
        for name, m in model.named_children():
            if name not in ("style_encoder", "predictor_encoder"):
                for p in m.parameters():
                    p.requires_grad_(False)
    elif mode == "diffusion_only":
        for name, m in model.named_children():
            if name != "diffusion":
                for p in m.parameters():
                    p.requires_grad_(False)
    elif mode == "no_discriminators":
        for m in [model.mpd, model.msd, model.wd]:
            for p in m.parameters():
                p.requires_grad_(False)


if __name__ == "__main__":
    main()
