from collections import OrderedDict

import pytest

import gym
from gym import spaces

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions

import pytorch_lightning as pl

from lightning_baselines3.on_policy_models import A2C
from lightning_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike



class DummyModel(A2C):
    def __init__(self, optimizer, *args, **kwargs):
        super(DummyModel, self).__init__(*args, **kwargs)

        self.optimizer = optimizer

        if isinstance(self.action_space, spaces.Discrete):
            self.p = nn.Parameter(torch.ones(1, self.action_space.n) * 0.5)
        elif isinstance(self.action_space, spaces.Box):
            self.p = nn.Parameter(torch.ones(1, self.action_space.shape[0] * 2) * 0.5)
        else:
            raise Exception('Incompatible environment action space')

        self.save_hyperparameters()


    def forward(self, x, **kwargs):
        p = self.p.expand(x.shape[0], self.p.shape[-1])
        if isinstance(self.action_space, spaces.Discrete):
            dist = distributions.Categorical(probs=F.softmax(p, dim=1))
        elif isinstance(self.action_space, spaces.Box):
            p = torch.chunk(p, 2, dim=1)
            dist = distributions.Normal(loc=p[0], scale=p[1])
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
                out = distributions.Normal(loc=p[0], scale=p[1]).sample()
        return out.cpu().numpy()


    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(params=self.parameters(), lr=1e-3)
        elif self.optimizer == 'RMSpropTFLike':
            return RMSpropTFLike(params=self.parameters(), lr=1e-3, momentum=0.9, centered=True)
        else:
            raise



@pytest.mark.parametrize("env_id", ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0"])
@pytest.mark.parametrize("optimizer", ['adam', 'RMSpropTFLike'])
def test_a2c_model(env_id, optimizer):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    model = DummyModel(
        optimizer=optimizer,
        env=env_id,
        eval_env=env_id,
        num_rollouts=10,
        num_eval_episodes=5,
        use_sde=False,
        sde_sample_freq=-1,
        seed=1234)

    trainer = pl.Trainer(max_epochs=2, terminate_on_nan=True)
    trainer.fit(model)
