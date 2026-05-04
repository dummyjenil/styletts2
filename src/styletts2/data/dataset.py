import os.path as osp

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from styletts2.data.text import TextCleaner


class StyleTTS2Dataset(Dataset):
    def __init__(
        self,
        data_list: list[str],
        root_path: str,
        sr: int = 24000,
        ood_data: str = "Data/OOD_texts.txt",
        min_length: int = 50,
        n_mels: int = 80,
        n_fft: int = 2048,
        win_length: int = 1200,
        hop_length: int = 300,
        max_mel_length: int = 192,
    ):
        # Parse data list: wave_path|text|speaker_id
        _data_list = [line.strip().split("|") for line in data_list if line.strip()]
        self.data_list = []
        for data in _data_list:
            if len(data) == 3:
                self.data_list.append(data)
            elif len(data) == 2:
                self.data_list.append((*data, "0"))
            else:
                continue  # Skip invalid lines

        self.text_cleaner = TextCleaner()
        self.sr = sr
        self.df = pd.DataFrame(self.data_list)
        self.root_path = root_path
        self.min_length = min_length
        self.max_mel_length = max_mel_length

        # Audio params
        self.n_mels = n_mels
        self.mean, self.std = -4, 4
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        # Load OOD texts
        self.ptexts = []
        if osp.exists(ood_data):
            with open(ood_data, encoding="utf-8") as f:
                tl = f.readlines()
            if tl:
                # Try to detect if it's a metadata file or plain text
                idx = 1 if len(tl[0].split("|")) > 1 else 0
                self.ptexts = [t.split("|")[idx].strip() for t in tl if t.strip()]

        if not self.ptexts:
            self.ptexts = [
                "This is a fallback dummy text for out of distribution training."
            ]

    def preprocess(self, wave: np.ndarray):
        wave_tensor = torch.from_numpy(wave).float()
        if wave_tensor.ndim == 1:
            wave_tensor = wave_tensor.unsqueeze(0)
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def __len__(self):
        return len(self.data_list)

    def _load_tensor(self, data: list[str]):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        full_path = osp.join(self.root_path, wave_path)

        try:
            wave, sr = sf.read(full_path)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            # Return silence as fallback
            wave = np.zeros(self.sr * 2)
            sr = self.sr

        if wave.ndim == 2:
            wave = wave[:, 0]

        if sr != self.sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr)

        # Padding silence (approx 200ms)
        pad_len = 5000
        wave = np.concatenate([np.zeros([pad_len]), wave, np.zeros([pad_len])], axis=0)

        text_ids = self.text_cleaner(text)
        text_ids = [0, *text_ids, 0]  # Add SOS/EOS (assuming 0 is pad/sos/eos)
        text_tensor = torch.LongTensor(text_ids)

        return wave, text_tensor, speaker_id

    def _load_data(self, data: list[str]):
        wave, _, speaker_id = self._load_tensor(data)
        mel_tensor = self.preprocess(wave).squeeze(0)

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[
                :, random_start : random_start + self.max_mel_length
            ]

        return mel_tensor, speaker_id

    def __getitem__(self, index: int):
        data = self.data_list[index]
        path = data[0]

        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = self.preprocess(wave).squeeze(0)

        # Ensure even length for model compatibility (e.g. downsampling)
        length_feature = mel_tensor.size(1)
        mel_tensor = mel_tensor[:, : (length_feature - length_feature % 2)]

        # Reference sample from same speaker
        # Filter df to find the same speaker and sample
        speaker_samples = self.df[self.df[2] == str(data[2])]
        ref_data = speaker_samples.sample(n=1).iloc[0].tolist()
        ref_mel_tensor, _ = self._load_data(ref_data)

        # OOD text for prosody/diffusion training
        ref_text_tensor = None
        max_tries = 10
        for _ in range(max_tries):
            rand_idx = np.random.randint(0, len(self.ptexts))
            ps = self.ptexts[rand_idx]
            if len(ps) >= self.min_length or _ == max_tries - 1:
                text_ids = self.text_cleaner(ps)
                text_ids = [0, *text_ids, 0]
                ref_text_tensor = torch.LongTensor(text_ids)
                break

        return (
            speaker_id,
            mel_tensor,
            text_tensor,
            ref_text_tensor,
            ref_mel_tensor,
            path,
            wave,
        )


class Collater:
    def __init__(self, return_wave: bool = False, max_mel_length: int = 192):
        self.return_wave = return_wave
        self.max_mel_length = max_mel_length

    def __call__(self, batch):
        # batch: List[Tuple(speaker_id, mel, text, ref_text, ref_mel, path, wave)]
        # Sort by mel length (descending) to help with RNN/Padding
        batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
        batch_size = len(batch)

        nmels = batch[0][1].size(0)
        max_mel_len = max([b[1].shape[1] for b in batch])
        max_text_len = max([b[2].shape[0] for b in batch])
        max_rtext_len = max([b[3].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_len)).float()
        texts = torch.zeros((batch_size, max_text_len)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_len)).long()
        labels = torch.zeros(batch_size).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()

        # ref_mels are often fixed length for the style encoder
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()

        waves = []
        paths = []

        for bid, (speaker_id, mel, text, ref_text, ref_mel, path, wave) in enumerate(
            batch
        ):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)

            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            labels[bid] = speaker_id

            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size

            ref_mel_size = ref_mel.size(1)
            # Clip if ref_mel is longer than max_mel_length (shouldn't happen with preprocess crop)
            ref_mel_size = min(ref_mel_size, self.max_mel_length)
            ref_mels[bid, :, :ref_mel_size] = ref_mel[:, :ref_mel_size]

            if self.return_wave:
                waves.append(wave)
            paths.append(path)

        if self.return_wave:
            return (
                texts,
                input_lengths,
                ref_texts,
                ref_lengths,
                mels,
                output_lengths,
                ref_mels,
                labels,
                waves,
            )

        return (
            texts,
            input_lengths,
            ref_texts,
            ref_lengths,
            mels,
            output_lengths,
            ref_mels,
            labels,
        )


def build_dataloader(
    path_list: list[str],
    root_path: str,
    batch_size: int = 4,
    num_workers: int = 1,
    device: str = "cpu",
    validation: bool = False,
    **kwargs,
):
    max_mel_length = kwargs.get("max_mel_length", 192)
    dataset = StyleTTS2Dataset(path_list, root_path, **kwargs)
    collater = Collater(max_mel_length=max_mel_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collater,
        pin_memory=(device != "cpu"),
    )
