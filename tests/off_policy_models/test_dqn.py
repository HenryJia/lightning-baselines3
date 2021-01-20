from collections import OrderedDict

import pytest

import gym
from gym import spaces

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions

import pytorch_lightning as pl

from lightning_baselines3.off_policy_models import DQN



class DummyModel(DQN):
    def __init__(self, *args, **kwargs):
        super(DummyModel, self).__init__(*args, **kwargs)
        self.p = nn.Parameter(torch.ones(1, self.action_space.n))
        self.p_target = nn.Parameter(torch.ones(1, self.action_space.n))
        self.save_hyperparameters()

    def forward(self, x, **kwargs):
        p = self.p.expand(x.shape[0], self.p.shape[-1])
        return p

    def forward_target(self, x):
        p_target = self.p_target.expand(x.shape[0], self.p.shape[-1])
        return p_target

    def update_target(self):
        self.p_target.data = self.p.data.clone()

    def predict(self, x, deterministic=True):
        p = self.p.expand(x.shape[0], self.p.shape[-1])
        if deterministic:
            out = torch.max(p, dim=1)[1]
        else:
            out = distributions.Categorical(probs=F.softmax(p, dim=1)).sample()
        return out.cpu().numpy()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



@pytest.mark.parametrize("env_id", ["CartPole-v1", "MountainCar-v0"])
def test_off_dqn_model(env_id):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    model = DummyModel(
        env_id,
        eval_env=env_id,
        batch_size=256,
        buffer_length=1000,
        warmup_length=100,
        train_freq=500,
        num_rollouts=10,
        num_eval_episodes=10,
        gamma=0.9,
        monitor_wrapper=True,
        seed=None)

    trainer = pl.Trainer(max_epochs=5, terminate_on_nan=True)
    trainer.fit(model)
