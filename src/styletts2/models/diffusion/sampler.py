import torch
from torch import nn


class LogNormalDistribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, num_samples, device):
        return (self.mean + self.std * torch.randn(num_samples, device=device)).exp()


class KarrasSchedule:
    def __init__(self, sigma_min, sigma_max, rho):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def __call__(self, num_steps, device):
        inv_rho = 1 / self.rho
        steps = torch.arange(num_steps, device=device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        return torch.cat([sigmas, sigmas.new_zeros([1])])


class AudioDiffusionConditional(nn.Module):
    def __init__(
        self,
        in_channels,
        embedding_max_length,
        embedding_features,
        embedding_mask_proba,
        channels,
        context_features,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_mask_proba = embedding_mask_proba
        self.diffusion = None  # Will be set in main model
        self.unet = None  # Will be set in main model

    def forward(self, x, **kwargs):
        if self.diffusion is None:
            raise RuntimeError("Diffusion model not set in AudioDiffusionConditional")
        return self.diffusion(x, **kwargs)
