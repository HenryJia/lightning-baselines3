import torch
from torch.distributions import (
    Distribution,
    MultivariateNormal
)

class SquashedMultivariateNormal(MultivariateNormal):
    """
    Gaussian distribution followed by a squashing function (tanh) to ensure bounds.
    """

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, -1 + 1e-6, 1 - 1e-6)
        # Log likelihood for Gaussian with change of variable
        log_prob = super().log_prob(torch.atanh(value))
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= torch.sum(torch.log(1 - value ** 2), dim=1)
        return log_prob

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.tanh(super().rsample())

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(super().mean())

    @property
    def variance(self) -> torch.Tensor:
        raise NotImplementedError

    def entropy(self) -> torch.Tensor:
        raise NotImplementedError
