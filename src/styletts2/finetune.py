import logging
import os

import torch

from styletts2.config import StyleTTS2Config
from styletts2.data.dataset import build_dataloader
from styletts2.models.styletts2 import StyleTTS2Model
from styletts2.trainer.trainer import StyleTTS2Trainer
from styletts2.utils.helpers import get_data_path_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Direct Configuration (No need for external YAML if defaults are enough)
    # You can override parameters here if needed
    config = StyleTTS2Config()

    # Example of overriding config in-script:
    # config.batch_size = 16
    # config.data_params.root_path = "path/to/data"

    os.makedirs(config.log_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Loaders
    train_list, val_list = get_data_path_list(
        config.data_params.train_data, config.data_params.val_data
    )

    train_dataloader = build_dataloader(
        train_list,
        config.data_params.root_path,
        batch_size=config.batch_size,
        num_workers=4,
        device=device,
        OOD_data=config.data_params.OOD_data,
        min_length=config.data_params.min_length,
        n_mels=config.model_params.n_mels,
    )

    val_dataloader = build_dataloader(
        val_list,
        config.data_params.root_path,
        batch_size=config.batch_size,
        num_workers=0,
        device=device,
        validation=True,
        OOD_data=config.data_params.OOD_data,
        min_length=config.data_params.min_length,
        n_mels=config.model_params.n_mels,
    )

    # Build Model
    if config.pretrained_model:
        logger.info(f"Loading pretrained model from {config.pretrained_model}")
        model = StyleTTS2Model.from_pretrained(config.pretrained_model, config)
    else:
        model = StyleTTS2Model(config)

    # Trainer
    trainer = StyleTTS2Trainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
    )

    # Start Training
    trainer.train()


if __name__ == "__main__":
    main()
