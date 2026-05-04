from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor, nn

from styletts2.models.diffusion.utils import default


def to_batch(
    x: Optional[float], xs: Optional[Tensor], batch_size: int, device: torch.device
) -> Tensor:
    if xs is not None:
        return xs
    if x is None:
        raise ValueError("Either x or xs must be provided")
    return torch.full((batch_size,), x, device=device)


class KarrasSampler:
    def __init__(self, s_tmin=0, s_tmax=float("inf"), s_churn=0.0, s_noise=1.0):
        self.params = (s_tmin, s_tmax, s_churn, s_noise)

    def step(self, x, fn, sigma, sigma_next, gamma):
        _s_tmin, _s_tmax, _, s_noise = self.params
        sigma_hat = sigma * (1 + gamma)
        x_hat = x + sqrt(max(sigma_hat**2 - sigma**2, 0)) * s_noise * torch.randn_like(
            x
        )
        d = (x_hat - fn(x_hat, sigma=sigma_hat)) / sigma_hat
        x_next = x_hat + (sigma_next - sigma_hat) * d
        if sigma_next != 0:
            d_prime = (x_next - fn(x_next, sigma=sigma_next)) / sigma_next
            x_next = x_hat + 0.5 * (sigma - sigma_hat) * (d + d_prime)
        return x_next

    def forward(self, noise, fn, sigmas, num_steps):
        s_tmin, s_tmax, s_churn, _ = self.params
        x, gammas = (
            sigmas[0] * noise,
            torch.where(
                (sigmas >= s_tmin) & (sigmas <= s_tmax),
                min(s_churn / num_steps, sqrt(2) - 1),
                0.0,
            ),
        )
        for i in range(num_steps - 1):
            x = self.step(x, fn, sigmas[i], sigmas[i + 1], gammas[i])
        return x


class ADPM2Sampler:
    def __init__(self, rho=1.0):
        self.rho = rho

    def step(self, x, fn, sigma, sigma_next):
        r = self.rho
        s_up = (
            sqrt(sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2)
            if sigma > sigma_next
            else 0
        )
        s_down = sqrt(max(sigma_next**2 - s_up**2, 0))
        s_mid = ((sigma ** (1 / r) + s_down ** (1 / r)) / 2) ** r

        d = (x - fn(x, sigma=sigma)) / sigma
        x_mid = x + d * (s_mid - sigma)
        d_mid = (x_mid - fn(x_mid, sigma=s_mid)) / s_mid
        x_next = x + d_mid * (s_down - sigma) + torch.randn_like(x) * s_up
        return x_next

    def forward(self, noise, fn, sigmas, num_steps):
        x = sigmas[0] * noise
        for i in range(num_steps - 1):
            x = self.step(x, fn, sigmas[i], sigmas[i + 1])
        return x


class DiffusionSampler(nn.Module):
    def __init__(
        self, diffusion, *, sampler, sigma_schedule, num_steps=None, clamp=True
    ):
        super().__init__()
        self.diffusion, self.sampler, self.sigma_schedule = (
            diffusion,
            sampler,
            sigma_schedule,
        )
        self.num_steps, self.clamp = num_steps, clamp

    def forward(self, noise, num_steps=None, **kwargs):
        num_steps = default(num_steps, self.num_steps)
        sigmas = self.sigma_schedule(num_steps, noise.device)

        def fn(x, sigma):
            return self.diffusion.denoise_fn(x, sigma=sigma, **kwargs)

        x = self.sampler(noise, fn, sigmas, num_steps)
        return x.clamp(-1.0, 1.0) if self.clamp else x


class KDiffusion:
    alias = "k"

    def __init__(self, net, *, sigma_distribution, sigma_data, dynamic_threshold=0.0):
        self.net, self.sigma_data = net, sigma_data
        self.dist, self.threshold = sigma_distribution, dynamic_threshold

    def get_scales(self, sigmas):
        s_data = self.sigma_data
        sigmas = rearrange(sigmas, "b -> b 1 1")
        return (
            s_data**2 / (sigmas**2 + s_data**2),
            sigmas * s_data * (s_data**2 + sigmas**2) ** -0.5,
            (sigmas**2 + s_data**2) ** -0.5,
            torch.log(sigmas) * 0.25,
        )

    def denoise_fn(self, x_noisy, sigmas=None, sigma=None, **kwargs):
        sigmas = to_batch(sigma, sigmas, x_noisy.shape[0], x_noisy.device)
        c_skip, c_out, c_in, c_noise = self.get_scales(sigmas)
        return c_skip * x_noisy + c_out * self.net(
            c_in * x_noisy, c_noise.squeeze(), **kwargs
        )

    def forward(self, x, **kwargs):
        sigmas = self.dist(x.shape[0], x.device)
        noise = torch.randn_like(x)
        x_noisy = x + rearrange(sigmas, "b -> b 1 1") * noise
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)
        weight = (sigmas**2 + self.sigma_data**2) * (sigmas * self.sigma_data) ** -2
        return (
            reduce(F.mse_loss(x_denoised, x, reduction="none"), "b ... -> b", "mean")
            * weight
        ).mean()
