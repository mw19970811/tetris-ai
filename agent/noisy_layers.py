"""Noisy Networks for exploration: NoisyLinear layer with factorised Gaussian noise.

Replaces epsilon-greedy with learned, state-dependent exploration.
Reference: Fortunato et al. (2018), "Noisy Networks for Exploration"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Linear layer with factorised Gaussian noise on weights and biases.

    y = (b_mu + b_sigma ⊙ ε_b) + (W_mu + W_sigma ⊙ ε_w) x

    Factorised noise reduces parameters from O(in×out) to O(in+out).
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.reset_parameters(sigma_init)
        self._noise_in = None
        self._noise_out = None
        self._eps_w = None
        self._eps_b = None

    def reset_parameters(self, sigma_init: float = 0.017):
        # Mu: uniform initialisation as in standard linear layer.
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        # Sigma: initialised to global_scale / sqrt(in_features).
        nn.init.constant_(self.weight_sigma, sigma_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, sigma_init / math.sqrt(self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Sample new noise each forward pass during training.
            self._sample_noise(x.device)
            weight = self.weight_mu + self.weight_sigma * self._eps_w
            bias = self.bias_mu + self.bias_sigma * self._eps_b
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def _sample_noise(self, device):
        # Factorised Gaussian noise: ε_w = f(ε_in) ⊗ f(ε_out)
        eps_in = self._f(torch.randn(self.in_features, device=device))
        eps_out = self._f(torch.randn(self.out_features, device=device))
        self._eps_w = torch.outer(eps_out, eps_in)
        self._eps_b = eps_out

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """f(x) = sign(x) * sqrt(|x|)"""
        return x.sign() * x.abs().sqrt()

    def scale_sigma(self, factor: float):
        """Multiply all sigma parameters by *factor* (for scheduled decay)."""
        with torch.no_grad():
            self.weight_sigma.mul_(factor)
            self.bias_sigma.mul_(factor)

    def set_sigma_scale(self, target_sigma: float):
        """Reset sigma parameters to *target_sigma* (global scale).

        Each parameter is set to target_sigma / sqrt(in_features), matching
        the NoisyLinear initialisation convention.  Previous learned sigma
        structure is overwritten — this is intentional for sawtooth schedules.
        """
        per_param = target_sigma / math.sqrt(self.in_features)
        with torch.no_grad():
            self.weight_sigma.fill_(per_param)
            self.bias_sigma.fill_(per_param)

    def get_sigma_mean(self) -> float:
        """Return the mean absolute sigma value (for logging)."""
        return float(self.weight_sigma.abs().mean().item())

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


def replace_linear_with_noisy(module: nn.Module, sigma_init: float = 0.017,
                               target_types=(nn.Linear,)) -> nn.Module:
    """Recursively replace nn.Linear layers with NoisyLinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, target_types):
            setattr(module, name, NoisyLinear(child.in_features, child.out_features, sigma_init))
        else:
            replace_linear_with_noisy(child, sigma_init, target_types)
    return module
