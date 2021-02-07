import copy
from collections import OrderedDict

import pytest

import gym
from gym import spaces

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions

import pytorch_lightning as pl

from lightning_baselines3.off_policy_models import SAC
from lightning_baselines3.common.utils import polyak_update



class DummyModel(SAC):
    def __init__(self, *args, **kwargs):
        super(DummyModel, self).__init__(*args, **kwargs)
        self.actor = nn.Linear(self.observation_space.shape[0], 2 * self.action_space.shape[0])
        self.critic1 = nn.Linear(self.observation_space.shape[0] + self.action_space.shape[0], 1)
        self.critic2 = nn.Linear(self.observation_space.shape[0] + self.action_space.shape[0], 1)

        self.critic_target1 = copy.deepcopy(self.critic1)
        self.critic_target2 = copy.deepcopy(self.critic2)

        self.save_hyperparameters()

    def forward_actor(self, x):
        out = list(torch.chunk(self.actor(x), 2, dim=1))
        out[1] = torch.diag_embed(
            torch.exp(0.5 * torch.clamp(out[1], -5, 5)))
        dist = distributions.MultivariateNormal(
            loc=out[0], scale_tril=out[1])
        return dist

    def forward_critics(self, obs, action):
        out = [
            self.critic1(torch.cat([obs, action], dim=1)),
            self.critic2(torch.cat([obs, action], dim=1))]
        return out

    def forward_critic_targets(self, obs, action):
        out = [
            self.critic_target1(torch.cat([obs, action], dim=1)),
            self.critic_target1(torch.cat([obs, action], dim=1))]
        return out

    def update_targets(self):
        polyak_update(self.critic1.parameters(), self.critic_target1.parameters(), tau=0.005)
        polyak_update(self.critic2.parameters(), self.critic_target2.parameters(), tau=0.005)

    def predict(self, x, deterministic=True):
        out = self.actor(x)
        if deterministic:
            out = torch.chunk(out, 2, dim=1)[0]
        else:
            out = list(torch.chunk(out, 2, dim=1))
            out[1] = torch.diag_embed(
                torch.exp(0.5 * torch.clamp(out[1], -5, 5)))
            dist = distributions.MultivariateNormal(
                loc=out[0], scale_tril=out[1])
        return dist.cpu().numpy()

    def configure_optimizers(self):
        opt_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        opt_critic = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=1e-3)
        return opt_critic, opt_actor


@pytest.mark.parametrize("env_id", ["MountainCarContinuous-v0", "LunarLanderContinuous-v2"])
def test_off_sac_model(env_id):
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
        num_rollouts=1,
        num_eval_episodes=10,
        gamma=0.9,
        seed=1234)

    trainer = pl.Trainer(max_epochs=2, terminate_on_nan=True)
    trainer.fit(model)
