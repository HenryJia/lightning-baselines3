import copy

import numpy as np

import torch
from torch import nn

import pytorch_lightning as pl

from lightning_baselines3.off_policy_models import SAC
from lightning_baselines3.common.distributions import SquashedMultivariateNormal
from lightning_baselines3.common.utils import polyak_update


class Model(SAC):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs, squashed_actions=True)

        # Note: The output layer of the actor must be Tanh activated
        self.actor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space.shape[0] * 2))

        in_dim = self.observation_space.shape[0] + self.action_space.shape[0]
        self.critic1 = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

        self.critic2 = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

        self.critic_target1 = copy.deepcopy(self.critic1)
        self.critic_target2 = copy.deepcopy(self.critic2)

        self.save_hyperparameters()

    def forward_actor(self, x):
        out = list(torch.chunk(self.actor(x), 2, dim=1))
        out[1] = torch.diag_embed(
            torch.exp(torch.clamp(out[1], -5, 5)))
        dist = SquashedMultivariateNormal(
            loc=torch.tanh(out[0]), scale_tril=out[1])
        return dist

    def forward_critics(self, obs, action):
        out = [
            self.critic1(torch.cat([obs, action], dim=1)),
            self.critic2(torch.cat([obs, action], dim=1))]
        return out

    def forward_critic_targets(self, obs, action):
        out = [
            self.critic_target1(torch.cat([obs, action], dim=1)),
            self.critic_target2(torch.cat([obs, action], dim=1))]
        return out

    def update_targets(self):
        polyak_update(
            self.critic1.parameters(),
            self.critic_target1.parameters(),
            tau=0.005)
        polyak_update(
            self.critic2.parameters(),
            self.critic_target2.parameters(),
            tau=0.005)

    def predict(self, x, deterministic=True):
        out = self.actor(x)
        if deterministic:
            out = torch.chunk(out, 2, dim=1)[0]
        else:
            out = list(torch.chunk(out, 2, dim=1))
            out[1] = torch.diag_embed(
                torch.exp(torch.clamp(out[1], -5, 5)))
            out = SquashedMultivariateNormal(
                loc=torch.tanh(out[0]), scale_tril=out[1]).sample()
        return out.cpu().numpy()

    def configure_optimizers(self):
        opt_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        opt_critic = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=3e-4)
        return opt_critic, opt_actor


if __name__ == '__main__':
    model = Model(
        env='LunarLanderContinuous-v2',
        eval_env='LunarLanderContinuous-v2',
        warmup_length=1000)

    trainer = pl.Trainer(max_epochs=20, gradient_clip_val=0.5)
    trainer.fit(model)

    model.evaluate(num_eval_episodes=10, render=True)
