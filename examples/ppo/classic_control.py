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

        net = [nn.Linear(self.observation_space.shape[0], hidden_size)]
        if isinstance(self.action_space, spaces.Discrete):
            net += [nn.Linear(hidden_size, self.action_space.n + 1)]
        elif isinstance(self.action_space, spaces.Box2D):
            net += [nn.Linear(hidden_size, self.action_space.shape[0] + 1)]
        else:
            raise Exception("This example only supports environments with Discrete and Box2D action spaces")

        self.net = nn.Sequential(*net)

    def forward(self, x, **kwargs):
        out = self.net(x)
        out, value = out[:, :-1], out[:, -1]
        if isinstance(self.action_space, spaces.Discrete):
            dist = distributions.Categorical(probs=F.softmax(out, dim=1))
        elif isinstance(self.action_space, spaces.Box2D):
            out = torch.chunk(out, 2, dim=1)
            dist = distributions.Normal(loc=out[0], scale=out[1])
        return dist, value


    def predict(self, x, deterministic=True):
        out = self.net(x)[:, :-1]
        if deterministic:
            if isinstance(self.action_space, spaces.Discrete):
                out = torch.max(out, dim=1)[1]
            elif isinstance(self.action_space, spaces.Box2D):
                out = torch.chunk(out, 2, dim=1)[0]
        else:
            if isinstance(self.action_space, spaces.Discrete):
                out = distributions.Categorical(probs=out).sample()
            elif isinstance(self.action_space, spaces.Box2D):
                out = torch.chunk(out, 2, dim=1)
                out = distributions.Normal(loc=out[0], scale=out[1]).sample()
        return out.cpu().numpy()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


env = make_vec_env('CartPole-v1', n_envs=4)

model = Model(
    env=env,
    eval_env = gym.make('CartPole-v1'),
    buffer_length=2048,
    num_rollouts=1,
    batch_size=64,
    epochs_per_rollout=10,
    gamma=0.99,
    gae_lambda=0.95,
    num_eval_episodes=100,
    use_sde=False,
    sde_sample_freq=-1,
    monitor_wrapper=True,
    seed=None)

trainer = pl.Trainer(max_epochs=3, terminate_on_nan=True, gpus=[0])
trainer.fit(model)

model.evaluate(deterministic=True, render=True)
