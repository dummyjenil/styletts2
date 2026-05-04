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


def main():
    parser = argparse.ArgumentParser(description="StyleTTS 2 Fine-tuning with PyTorch Lightning")
    parser.add_argument("--train_data", type=str, default=None, help="Path to training data (HF dataset)")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to pretrained model checkpoint")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs and checkpoints")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()

    config = StyleTTS2Config()
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.log_dir = args.log_dir

    # Data Module
    data_module = StyleTTS2DataModule(
        config=config,
        train_data=args.train_data,
        val_data=args.val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Build Model
    if args.pretrained_model or config.pretrained_model:
        ckpt_path = args.pretrained_model or config.pretrained_model
        logger.info(f"Loading pretrained model from {ckpt_path}")
        base_model = StyleTTS2Model.from_pretrained(ckpt_path, config)
    else:
        base_model = StyleTTS2Model(config)

    # Lightning Module
    model = StyleTTS2LightningModule(config=config, model=base_model)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "checkpoints"),
        filename="styletts2-{epoch:02d}-{train/mel_loss:.4f}",
        save_top_k=3,
        monitor="train/mel_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name="tensorboard")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=args.devices,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=config.log_interval,
        # StyleTTS2 uses manual optimization so we don't need precision='16-mixed' here unless we handle it in step
        # For now keeping it simple.
    )

    # Start Training
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
