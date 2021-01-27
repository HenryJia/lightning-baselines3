from collections import OrderedDict

import pytest

import gym
from gym import spaces

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions

import pytorch_lightning as pl

from lightning_baselines3.off_policy_models import TD3
from lightning_baselines3.common.utils import polyak_update



class DummyModel(TD3):
    def __init__(self, *args, **kwargs):
        super(DummyModel, self).__init__(*args, **kwargs)
        self.actor = nn.Linear(self.observation_space.shape[0], self.action_space.shape[0])
        self.actor_target = nn.Linear(self.observation_space.shape[0], self.action_space.shape[0])
        self.critic1 = nn.Linear(self.observation_space.shape[0] + self.action_space.shape[0], 1)
        self.critic2 = nn.Linear(self.observation_space.shape[0] + self.action_space.shape[0], 1)
        self.critic_target1 = nn.Linear(self.observation_space.shape[0] + self.action_space.shape[0], 1)
        self.critic_target2 = nn.Linear(self.observation_space.shape[0] + self.action_space.shape[0], 1)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        self.save_hyperparameters()

    def forward_actor(self, x):
        return self.actor(x)

    def forward_actor_target(self, x):
        return self.actor_target(x)

    def forward_critic1(self, obs, action):
        return self.critic1(torch.cat([obs, action], dim=1))

    def forward_critic2(self, obs, action):
        return self.critic2(torch.cat([obs, action], dim=1))

    def forward_critic_target1(self, obs, action):
        return self.critic_target1(torch.cat([obs, action], dim=1))

    def forward_critic_target2(self, obs, action):
        return self.critic_target2(torch.cat([obs, action], dim=1))

    def update_targets(self):
        polyak_update(self.actor.parameters(), self.actor_target.parameters(), tau=0.005)
        polyak_update(self.critic1.parameters(), self.critic_target1.parameters(), tau=0.005)
        polyak_update(self.critic2.parameters(), self.critic_target2.parameters(), tau=0.005)

    def predict(self, x, deterministic=True):
        out = self.actor(x)
        if not deterministic:
            out = out + torch.randn_like(out) * 0.1
        return out.cpu().numpy()

    def configure_optimizers(self):
        opt_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        opt_critic = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=1e-3)
        return opt_actor, opt_critic


@pytest.mark.parametrize("env_id", ["MountainCarContinuous-v0", "LunarLanderContinuous-v2"])
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
        num_rollouts=1,
        num_eval_episodes=10,
        gamma=0.9,
        seed=1234)

    trainer = pl.Trainer(max_epochs=2, terminate_on_nan=True)
    trainer.fit(model)
