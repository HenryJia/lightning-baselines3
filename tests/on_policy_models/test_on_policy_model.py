from collections import OrderedDict

import pytest

import gym
from gym import spaces

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions

import pytorch_lightning as pl

from lightning_baselines3.on_policy_models.on_policy_model import OnPolicyModel



class DummyModel(OnPolicyModel):
    def __init__(self, *args, **kwargs):
        super(DummyModel, self).__init__(*args, **kwargs)

        if isinstance(self.action_space, spaces.Discrete):
            self.p = nn.Parameter(torch.ones(1, self.action_space.n) * 0.5)
        elif isinstance(self.action_space, spaces.Box):
            self.p = nn.Parameter(torch.ones(1, self.action_space.shape[0] * 2) * 0.5)
        else:
            raise Exception('Incompatible environment action space')


    def forward(self, x, **kwargs):
        p = self.p.expand(x.shape[0], self.p.shape[-1])
        if isinstance(self.action_space, spaces.Discrete):
            dist = distributions.Categorical(probs=F.softmax(p, dim=1))
        elif isinstance(self.action_space, spaces.Box):
            p = torch.chunk(p, 2, dim=1)
            dist = distributions.Normal(loc=p[0], scale=1 + p[1] ** 2)
        return dist, torch.ones_like(x)[:, :1]


    def predict(self, x, deterministic=True):
        p = self.p.expand(x.shape[0], self.p.shape[-1])
        if deterministic:
            if isinstance(self.action_space, spaces.Discrete):
                out = torch.max(p, dim=1)[1]
            elif isinstance(self.action_space, spaces.Box):
                out = torch.chunk(p, 2, dim=1)[0]
        else:
            if isinstance(self.action_space, spaces.Discrete):
                out = distributions.Categorical(probs=F.softmax(p, dim=1)).sample()
            elif isinstance(self.action_space, spaces.Box):
                p = torch.chunk(p, 2, dim=1)
                out = distributions.Normal(loc=p[0], scale=1 + p[1] ** 2).sample()
        return out.cpu().numpy()


    def training_step(self, x, batch_idx):
        loss = self(x.observations)[0].entropy().mean()
        self.log('loss', loss)
        return loss 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




@pytest.mark.parametrize("env_id", ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0"])
def test_on_policy_model(env_id):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    model = DummyModel(
        env_id,
        eval_env=env_id,
        buffer_length=512,
        num_rollouts=1,
        batch_size=32,
        epochs_per_rollout=10,
        num_eval_episodes=10,
        gamma=0.9,
        gae_lambda=0.95,
        use_sde=False,
        sde_sample_freq=-1,
        verbose=1,
        seed=1234)

    trainer = pl.Trainer(max_epochs=2, terminate_on_nan=True)
    trainer.fit(model)
