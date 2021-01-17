from collections import OrderedDict
import argparse
import copy

import numpy as np
import gym
from gym import spaces
import pybullet_envs

import torch
from  torch import distributions
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

from lightning_baselines3.off_policy_models import DQN
from lightning_baselines3.common.vec_env import make_vec_env, SubprocVecEnv



class Model(DQN):
    def __init__(self, lr=3e-4, hidden_size=64, eps_init=1.0, eps_decay=10000, eps_final=0.05, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.lr = lr

        self.qnet = nn.Sequential(
            nn.BatchNorm1d(self.observation_space.shape[0]),
            nn.Linear(self.observation_space.shape[0], hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.action_space.n))

        self.eps = eps_init
        self.eps_init = eps_init
        self.eps_decay = eps_decay
        self.eps_final = eps_final

        self.qnet_target = copy.deepcopy(self.qnet)

        self.save_hyperparameters()


    @staticmethod
    def add_model_specific_args(parent_parser=None):
        if parent_parser:
            parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        else:
            parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--hidden_size', type=int, default=64)
        parser.add_argument('--eps_init', type=float, default=1.0)
        parser.add_argument('--eps_decay', type=int, default=10000)
        parser.add_argument('--eps_final', type=float, default=0.05)
        parser.add_argument('--buffer_length', type=int, default=int(1e6))
        parser.add_argument('--warmup_length', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--train_freq', type=int, default=4)
        parser.add_argument('--num_rollouts', type=int, default=1024)
        parser.add_argument('--episodes_per_rollout', type=int, default=-1)
        parser.add_argument('--gradient_steps', type=int, default=1)
        parser.add_argument('--target_update_interval', type=int, default=512)
        parser.add_argument('--num_eval_episodes', type=int, default=10)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--monitor_wrapper', action='store_true')
        parser.add_argument('--seed', type=int)
        return parser


    def forward(self, x):
        return self.qnet(x)


    def forward_target(self, x):
        return self.qnet_target(x)


    def update_target(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


    def on_step(self): # Linearly decay our epsilon for epsilon greedy
        k = max(self.eps_decay - self.num_timesteps, 0) / self.eps_decay
        self.eps = self.eps_final + k * (self.eps_init - self.eps_final)


    def predict(self, x, deterministic=True):
        out = self.qnet(x)
        if deterministic:
            out = torch.max(out, dim=1)[1]
        else:
            eps = torch.rand_like(out[:, 0])
            eps = (eps < self.eps).float()
            out = eps * torch.max(torch.rand_like(out), dim=1)[1] + (1 - eps) * torch.max(out, dim=1)[1]
        return out.long().cpu().numpy()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        return optimizer



if __name__ == '__main__':
    # Parse env, model and trainer args separately so we don't have to abuse **kwargs
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--env', type=str)
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--evaluate', action='store_true') # If set to true, load model from model_fn and don't train
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

        env = make_vec_env(args.env, n_envs=args.num_env, vec_env_cls=SubprocVecEnv)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_reward_mean',
        dirpath=args.env+'_dqn_mlp',
        filename='mlp-{epoch:02d}-{val_reward_mean:.2f}.pl',
        save_top_k=1,
        mode='max',
        )

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
        model = Model.load_from_checkpoint(args.model_fn, env=env, eval_env=env)

        # Warning: for some reason PyBullet environments are hardcoded to record at 320x240, and there's no easy way to deal with this
        rewards, lengths = model.evaluate(num_eval_episodes=10, render=True, record=True, record_fn=args.video_fn)
        print('Mean rewards and length:', np.mean(rewards), np.mean(lengths))
