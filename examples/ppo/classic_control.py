from collections import OrderedDict
import argparse

import gym
from gym import spaces
import pybullet_envs

import torch
from  torch import distributions
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from lightning_baselines3.on_policy_models import PPO
from lightning_baselines3.common.vec_env import make_vec_env, SubprocVecEnv


class Model(PPO):
    def __init__(self, lr=3e-4, hidden_size=64, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.lr = lr

        actor = [
            nn.Linear(self.observation_space.shape[0], hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()]
        if isinstance(self.action_space, spaces.Discrete):
            actor += [
                nn.Linear(hidden_size, self.action_space.n),
                nn.Softmax(dim=1)]
        elif isinstance(self.action_space, spaces.Box):
            actor += [nn.Linear(hidden_size, self.action_space.shape[0] * 2)]
        else:
            raise Exception("This example only supports environments with Discrete and Box action spaces")

        self.actor = nn.Sequential(*actor)
        self.critic = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1))

        #self.save_hyperparameters('lr', 'hidden_size')

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        if parent_parser:
            parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        else:
            parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--hidden_size', type=int, default=64)
        parser.add_argument('--buffer_length', type=int, default=2048)
        parser.add_argument('--num_rollouts', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epochs_per_rollout', type=int, default=10)
        parser.add_argument('--num_eval_episodes', type=int, default=10)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--gae_lambda', type=float, default=0.97)
        parser.add_argument('--clip_range', type=float, default=0.2)
        parser.add_argument('--clip_range_vf', type=float)
        parser.add_argument('--value_coef', type=float, default=0.5)
        parser.add_argument('--entropy_coef', type=float, default=0.0)
        parser.add_argument('--use_sde', action='store_true')
        parser.add_argument('--sde_sample_freq', type=int, default=-1)
        parser.add_argument('--target_kl', type=float)
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--monitor_wrapper', action='store_true')
        parser.add_argument('--seed', type=int)
        return parser


    def forward(self, x):
        out = self.actor(x)
        if isinstance(self.action_space, spaces.Discrete):
            dist = distributions.Categorical(probs=out)
        elif isinstance(self.action_space, spaces.Box):
            out = list(torch.chunk(out, 2, dim=1))
            out[1] = torch.diag_embed(torch.exp(0.5 * torch.clamp(out[1], -5, 5)))
            dist = distributions.MultivariateNormal(loc=out[0], scale_tril=out[1])
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
                out = list(torch.chunk(out, 2, dim=1))
                out[1] = torch.diag_embed(torch.exp(0.5 * torch.clamp(out[1], -5, 5)))
                out = distributions.MultivariateNormal(loc=out[0], scale_tril=out[1]).sample()
        return out.cpu().numpy()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    # Parse env, model and trainer args separately so we don't have to abuse **kwargs
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument('--env', type=str)
    env_parser.add_argument('--num_env', type=int, default=4)
    env_args, ignored = env_parser.parse_known_args()

    model_parser = Model.add_model_specific_args()
    model_args, ignored = model_parser.parse_known_args()
    model_args = vars(model_args)

    trainer_parser = argparse.ArgumentParser(add_help=False)
    trainer_parser = pl.Trainer.add_argparse_args(trainer_parser)
    trainer_args, ignored = trainer_parser.parse_known_args()
    trainer_args = vars(trainer_args)

    env = make_vec_env(env_args.env, n_envs=env_args.num_env, vec_env_cls=SubprocVecEnv)

    model = Model(
        env=env,
        eval_env=gym.make(env_args.env),
        **model_args)

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model)

    model.evaluate(deterministic=True, render=True)
