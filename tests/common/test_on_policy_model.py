import pytest

import torch

from lightning_baselines3.common.on_policy_model import OnPolicyModel


class EmptyModel(OnPolicyModel):
    def forward(self, x):
        return torch.distributions.Bernoulli(probs=torch.zeros_like(x) + 0.5)

    def training_step(self, x):
        return 0


@pytest.mark.parametrize("env_id", ["CartPole-v0"])
def test_on_policy_model(env_id):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    model = EmptyModel(
        env_id,
        buffer_length=512,
        num_rollouts=1,
        batch_size=32,
        epochs_per_rollout=10,
        gamma=0.9,
        gae_lambda=0.95,
        ent_coef=0.1,
        vf_coef=1.0,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        buffer_grads=False,
        monitor_wrapper=True,
        seed=None)
