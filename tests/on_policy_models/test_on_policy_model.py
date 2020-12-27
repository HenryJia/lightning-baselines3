from collections import OrderedDict

import pytest

import torch
from torch import nn

import pytorch_lightning as pl

from lightning_baselines3.on_policy_models.on_policy_model import OnPolicyModel



class DummyModel(OnPolicyModel):
    def __init__(self, *args, **kwargs):
        super(DummyModel, self).__init__(*args, **kwargs)
        self.p = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, **kwargs):
        return torch.distributions.Bernoulli(probs=torch.zeros_like(x)[:, 0] + torch.clamp(self.p, 0, 1)), (self.p * 2).view(1, 1)

    def predict(self, x, deterministic=True):
        if deterministic:
            return torch.round(self.p).cpu().numpy()
        else:
            return torch.distributions.Bernoulli(probs=torch.zeros_like(x)[:, 0] + self.p).sample().cpu().numpy()

    def training_step(self, x, batch_idx):
        loss = self(x.observations)[0].entropy().mean()
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



@pytest.mark.parametrize("env_id", ["CartPole-v0"])
def test_on_policy_model(env_id):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    model = DummyModel(
        env_id,
        buffer_length=512,
        num_rollouts=1,
        batch_size=32,
        epochs_per_rollout=10,
        gamma=0.9,
        gae_lambda=0.95,
        use_sde=False,
        sde_sample_freq=-1,
        monitor_wrapper=True,
        seed=None)

    trainer = pl.Trainer(max_epochs=5, terminate_on_nan=True)
    trainer.fit(model)
