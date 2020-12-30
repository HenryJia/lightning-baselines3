from collections import OrderedDict

import gym
from gym import spaces

import torch
from  torch import distributions
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from lightning_baselines3.on_policy_models import PPO
from lightning_baselines3.common.vec_env import make_vec_env


class Model(PPO):
    def __init__(self, lr=3e-4, hidden_size=64, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.lr = lr

        actor = [nn.Linear(self.observation_space.shape[0], hidden_size)]
        if isinstance(self.action_space, spaces.Discrete):
            actor += [nn.Linear(hidden_size, self.action_space.n)]
        elif isinstance(self.action_space, spaces.Box):
            actor += [nn.Linear(hidden_size, self.action_space.shape[0])]
        else:
            raise Exception("This example only supports environments with Discrete and Box action spaces")

        self.actor = nn.Sequential(*actor)
        self.critic = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], hidden_size),
            nn.Linear(hidden_size, 1))


    def forward(self, x, **kwargs):
        out = self.actor(x)
        if isinstance(self.action_space, spaces.Discrete):
            dist = distributions.Categorical(probs=F.softmax(out, dim=1))
        elif isinstance(self.action_space, spaces.Box):
            out = torch.chunk(out, 2, dim=1)
            dist = distributions.Normal(loc=out[0], scale=out[1])
        return dist, self.critic(x).flatten()


    def predict(self, x, deterministic=True):
        out = self.actor(x)
        if deterministic:
            if isinstance(self.action_space, spaces.Discrete):
                out = torch.max(out, dim=1)[1]
            elif isinstance(self.action_space, spaces.Box):
                out = torch.chunk(out, 2, dim=1)[0]
        else:
            if isinstance(self.action_space, spaces.Discrete):
                out = distributions.Categorical(probs=out).sample()
            elif isinstance(self.action_space, spaces.Box):
                out = torch.chunk(out, 2, dim=1)
                out = distributions.Normal(loc=out[0], scale=out[1]).sample()
        return out.cpu().numpy()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


env = make_vec_env('CartPole-v1', n_envs=4)

model = Model(
    env=env,
    eval_env=gym.make('CartPole-v1'),
    num_eval_episodes=100,
    seed=1337)

trainer = pl.Trainer(max_epochs=4, terminate_on_nan=True, gpus=[0], gradient_clip_val=0.5)
trainer.fit(model)

model.evaluate(deterministic=True, render=True)
