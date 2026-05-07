import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger


class VoiceManager:
    """
    Manages loading and caching of voice packs (``.pt`` files) from disk.

    Expected tensor shape per voice: ``(N, style_dim * 2)`` where ``N`` is
    typically 400 for Kokoro-style packs (one row per token length).

    Parameters
    ----------
    voices_dir:
        Directory that contains ``<name>.pt`` voice pack files.
    device:
        Target torch device string (``"cpu"``, ``"cuda"``, …).
    cache_size:
        Maximum number of voice tensors to keep in memory simultaneously.
        Oldest entry is evicted when the cache is full.  Defaults to 6
        (matching the base model's number of speakers).
    """

    def __init__(
        self,
        voices_dir: Optional[str] = None,
        device: str = "cpu",
        cache_size: int = 6,
    ) -> None:
        self.voices_dir = voices_dir
        self.device = device
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_size = cache_size

        if voices_dir and not os.path.exists(voices_dir):
            logger.warning(f"Voices directory not found: {voices_dir}")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_voice(self, voice_name: str) -> torch.Tensor:
        """
        Fetch a voice tensor by name, loading from disk on first access.

        The tensor is cached in memory after the first load.  When the cache
        is full the oldest entry is evicted (FIFO).

        Args:
            voice_name: Bare name (e.g. ``"en_female_1"``) or an absolute
                        path to a ``.pt`` file.

        Returns:
            A float tensor on the configured ``device``.

        Raises:
            ValueError: If no ``voices_dir`` was set.
            FileNotFoundError: If the voice file cannot be found.
        """
        # Fast path — already cached
        if voice_name in self._cache:
            logger.debug(f"Voice cache hit: {voice_name}")
            return self._cache[voice_name]

        # Resolve file path
        file_path = self._resolve_path(voice_name)

        logger.debug(f"Loading voice pack from disk: {file_path}")
        voice_tensor = torch.load(file_path, map_location=self.device, weights_only=True)
        voice_tensor = voice_tensor.to(self.device)

        # Evict oldest if full
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            logger.debug(f"Voice cache full — evicting: {oldest_key}")
            del self._cache[oldest_key]

        self._cache[voice_name] = voice_tensor
        return voice_tensor

    def list_voices(self) -> list[str]:
        """Return names of all ``.pt`` voice packs found in ``voices_dir``."""
        if not self.voices_dir or not os.path.exists(self.voices_dir):
            return []
        return sorted(p.stem for p in Path(self.voices_dir).glob("*.pt"))

    def save_voice(self, name: str, tensor: torch.Tensor) -> str:
        """
        Save a style tensor as a named voice pack and add it to the cache.

        Args:
            name: Voice name (filename without ``.pt``).
            tensor: Style tensor to save.

        Returns:
            Absolute path to the saved file.

        Raises:
            ValueError: If ``voices_dir`` is not set.
        """
        if not self.voices_dir:
            raise ValueError("voices_dir must be set to save a voice pack.")

        os.makedirs(self.voices_dir, exist_ok=True)
        file_path = os.path.join(self.voices_dir, f"{name}.pt")
        torch.save(tensor.cpu(), file_path)
        logger.info(f"Voice pack saved: {file_path}")

        # Update cache
        self._cache[name] = tensor.to(self.device)
        return file_path

    def clear_cache(self) -> None:
        """Evict all cached voice tensors."""
        self._cache.clear()
        logger.debug("Voice cache cleared.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_path(self, voice_name: str) -> str:
        """Resolve *voice_name* to an absolute file path."""
        # 1. Direct absolute path
        if os.path.isabs(voice_name) and os.path.exists(voice_name):
            return voice_name

        # 2. Name relative to voices_dir
        if self.voices_dir:
            candidate = os.path.join(self.voices_dir, f"{voice_name}.pt")
            if os.path.exists(candidate):
                return candidate

        # 3. Fallback: voice_name is itself a relative path
        if os.path.exists(voice_name):
            return voice_name

        raise FileNotFoundError(
            f"Voice pack '{voice_name}' not found. "
            f"Searched in voices_dir='{self.voices_dir}' and as a direct path."
        )
