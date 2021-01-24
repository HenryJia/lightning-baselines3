import argparse

import numpy as np
import gym
from gym import spaces
import pybullet_envs

import torch
from torch import distributions
from torch import nn

import pytorch_lightning as pl

from lightning_baselines3.on_policy_models import PPO
from lightning_baselines3.common.vec_env import make_vec_env, SubprocVecEnv
pybullet_envs.getList()


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
            raise Exception("This example only supports Discrete and Box")

        self.actor = nn.Sequential(*actor)
        self.critic = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1))

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        if parent_parser:
            parser = argparse.ArgumentParser(
                parents=[parent_parser], add_help=False)
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
        parser.add_argument('--seed', type=int)
        return parser

    def forward(self, x):
        out = self.actor(x)
        if isinstance(self.action_space, spaces.Discrete):
            dist = distributions.Categorical(probs=out)
        elif isinstance(self.action_space, spaces.Box):
            out = list(torch.chunk(out, 2, dim=1))
            out[1] = torch.diag_embed(
                torch.exp(0.5 * torch.clamp(out[1], -5, 5)))
            dist = distributions.MultivariateNormal(
                loc=out[0], scale_tril=out[1])
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
                out[1] = torch.diag_embed(
                    torch.exp(0.5 * torch.clamp(out[1], -5, 5)))
                out = distributions.MultivariateNormal(
                    loc=out[0], scale_tril=out[1]).sample()
        return out.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    # Parse args separately so we don't have to abuse **kwargs
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--env', type=str)
    parser.add_argument('--num_env', type=int, default=4)
    # If set to true, load model from model_fn and don't train
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--model_fn', type=str, default='ppo_mlp')
    parser.add_argument('--video_fn', type=str, default='ppo_mlp.mp4')
    args, ignored = parser.parse_known_args()

    if not args.evaluate:
        model_parser = Model.add_model_specific_args()
        model_args, ignored = model_parser.parse_known_args()
        model_args = vars(model_args)

        trainer_parser = argparse.ArgumentParser(add_help=False)
        trainer_parser = pl.Trainer.add_argparse_args(trainer_parser)
        trainer_args, ignored = trainer_parser.parse_known_args()
        trainer_args = vars(trainer_args)

        env = make_vec_env(
            args.env, n_envs=args.num_env, vec_env_cls=SubprocVecEnv)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_reward_mean',
            dirpath=args.env+'_ppo_mlp',
            filename='mlp-{epoch:02d}-{val_reward_mean:.2f}.pl',
            save_top_k=1,
            mode='max')

        model = Model(
            env=env,
            eval_env=gym.make(args.env),
            **model_args)

        trainer = pl.Trainer(**trainer_args, callbacks=[checkpoint_callback])
        trainer.fit(model)
    else:
        env = gym.make(args.env)
        if 'Bullet' in args.env:
            env.render(mode='human')
            env.reset()
        model = Model.load_from_checkpoint(
            args.model_fn, env=env, eval_env=env)

        # Warning: PyBullet environments are hardcoded to record at 320x240
        # There seems to be no easy way to deal with this
        rewards, lengths = model.evaluate(
            num_eval_episodes=10,
            render=True,
            record=True,
            record_fn=args.video_fn)
        print('Mean rewards and length:', np.mean(rewards), np.mean(lengths))
