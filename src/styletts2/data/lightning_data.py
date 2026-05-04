from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader

from styletts2.data.text import TextCleaner


class StyleTTS2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        train_data=None,
        val_data=None,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
    ):
        super().__init__()
        self.config = config
        self.train_data = train_data or config.data_params.train_data
        self.val_data = val_data or config.data_params.val_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.text_cleaner = TextCleaner()
        self.sr = config.preprocess_params.sr

        # Audio transform for mel
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_mels=config.model_params.n_mels,
            n_fft=2048,
            win_length=1200,
            hop_length=300,
        )
        self.mean, self.std = -4, 4

        self.train_dataset: Any = None
        self.val_dataset: Any = None
        self.speaker_to_indices = {}

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Load dataset using Hugging Face datasets
            # The user expects a dataset with 'audio', 'text', 'speaker' columns

            self.train_dataset = load_dataset(self.train_data, split="train")
            self.train_dataset = self.train_dataset.cast_column("audio", Audio(sampling_rate=self.sr))

            if self.val_data:
                self.val_dataset = load_dataset(self.val_data, split="train")
                self.val_dataset = self.val_dataset.cast_column("audio", Audio(sampling_rate=self.sr))

            # Pre-group indices by speaker for efficient reference sampling
            self.speaker_to_indices = {}
            for i, example in enumerate(self.train_dataset):
                sid = str(example['speaker'])
                if sid not in self.speaker_to_indices:
                    self.speaker_to_indices[sid] = []
                self.speaker_to_indices[sid].append(i)

    def train_dataloader(self):
        return DataLoader(
            cast(torch.utils.data.Dataset, self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self):
        if not self.val_dataset:
            return None
        return DataLoader(
            cast(torch.utils.data.Dataset, self.val_dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory
        )

    def preprocess_audio(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        if wave_tensor.ndim == 1:
            wave_tensor = wave_tensor.unsqueeze(0)
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def collate_fn(self, batch):
        processed_batch = []
        for item in batch:
            audio_array = item['audio']['array']
            text = item['text']
            speaker_id = item['speaker']

            # Padding silence (approx 200ms)
            pad_len = 5000
            wave = np.concatenate([np.zeros([pad_len]), audio_array, np.zeros([pad_len])], axis=0)

            # Text to IDs
            text_ids = self.text_cleaner(text)
            text_ids = [0, *text_ids, 0]
            text_tensor = torch.LongTensor(text_ids)

            # Mel
            mel_tensor = self.preprocess_audio(wave).squeeze(0)
            mel_tensor = mel_tensor[:, : (mel_tensor.size(1) - mel_tensor.size(1) % 2)]

            # Reference sample from same speaker
            sid_str = str(speaker_id)
            indices = self.speaker_to_indices.get(sid_str, [0])
            ref_idx = np.random.choice(indices)
            ref_item = self.train_dataset[int(ref_idx)]
            ref_audio = ref_item['audio']['array']
            ref_wave = np.concatenate([np.zeros([pad_len]), ref_audio, np.zeros([pad_len])], axis=0)
            ref_mel_tensor = self.preprocess_audio(ref_wave).squeeze(0)

            # Crop/Pad ref_mel to max_mel_length (192)
            max_mel_len = self.config.max_len // 2
            if ref_mel_tensor.size(1) > max_mel_len:
                start = np.random.randint(0, ref_mel_tensor.size(1) - max_mel_len)
                ref_mel_tensor = ref_mel_tensor[:, start : start + max_mel_len]

            processed_batch.append((
                int(speaker_id) if str(speaker_id).isdigit() else 0,
                mel_tensor,
                text_tensor,
                ref_mel_tensor,
                wave
            ))

        # Sort by mel length (descending) to help with RNN/Padding
        processed_batch = sorted(processed_batch, key=lambda x: x[1].shape[1], reverse=True)

        batch_size = len(processed_batch)
        nmels = processed_batch[0][1].size(0)
        max_mel_len = max([b[1].shape[1] for b in processed_batch])
        max_text_len = max([b[2].shape[0] for b in processed_batch])
        max_ref_mel_len = self.config.max_len // 2

        mels = torch.zeros((batch_size, nmels, max_mel_len)).float()
        texts = torch.zeros((batch_size, max_text_len)).long()
        ref_mels = torch.zeros((batch_size, nmels, max_ref_mel_len)).float()
        labels = torch.zeros(batch_size).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        waves = []

        for bid, (sid, mel, text, rmel, wave) in enumerate(processed_batch):
            mlen = mel.size(1)
            tlen = text.size(0)
            rmlen = rmel.size(1)

            mels[bid, :, :mlen] = mel
            texts[bid, :tlen] = text
            ref_mels[bid, :, :rmlen] = rmel
            labels[bid] = sid
            input_lengths[bid] = tlen
            output_lengths[bid] = mlen
            waves.append(wave)

        return (
            texts,
            input_lengths,
            mels,
            output_lengths,
            ref_mels,
            labels,
            waves,
        )
