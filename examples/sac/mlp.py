import argparse
import copy

import numpy as np
import gym
import pybullet_envs

import torch
from torch import nn

import pytorch_lightning as pl

from lightning_baselines3.off_policy_models import SAC
from lightning_baselines3.common.utils import polyak_update
pybullet_envs.getList()


class Model(SAC):
    def __init__(self, hidden_size, lr, tau, action_noise, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        self.lr = lr
        self.tau = tau
        self.action_noise = action_noise

        # Note: The output layer of the actor must be Tanh activated
        self.actor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.action_space.shape[0] * 2))

        in_dim = self.observation_space.shape[0] + self.action_space.shape[0]
        self.critic1 = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1))

        self.critic2 = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1))

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
        polyak_update(
            self.critic1.parameters(),
            self.critic_target1.parameters(),
            tau=self.tau)
        polyak_update(
            self.critic2.parameters(),
            self.critic_target2.parameters(),
            tau=self.tau)

    def predict(self, x, deterministic=True):
        out = self.actor(x)
        if deterministic:
            out = torch.chunk(out, 2, dim=1)[0]
        else:
            out = list(torch.chunk(out, 2, dim=1))
            out[1] = torch.diag_embed(
                torch.exp(0.5 * torch.clamp(out[1], -5, 5)))
            out = distributions.MultivariateNormal(
                loc=out[0], scale_tril=out[1]).sample()
        return out.cpu().numpy()

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        if parent_parser:
            parser = argparse.ArgumentParser(
                parents=[parent_parser], add_help=False)
        else:
            parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--hidden_size', type=int, default=64)
        parser.add_argument('--tau', type=float, default=0.005)
        parser.add_argument('--buffer_length', type=int, default=int(1e6))
        parser.add_argument('--warmup_length', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--train_freq', type=int, default=1)
        parser.add_argument('--num_rollouts', type=int, default=1000)
        parser.add_argument('--episodes_per_rollout', type=int, default=-1)
        parser.add_argument('--gradient_steps', type=int, default=1)
        parser.add_argument('--target_policy_noise', type=float, default=0.2)
        parser.add_argument('--target_noise_clip', type=float, default=0.5)
        parser.add_argument('--num_eval_episodes', type=int, default=10)
        parser.add_argument('--target_update_interval', type=int, default=1)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--entropy_coef', default='auto')
        parser.add_argument('--target_entropy', default='auto')
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--use_sde', action='store_true')
        parser.add_argument('--sde_sample_freq', type=int, default=-1)
        parser.add_argument('--use_sde_at_warmup', action='store_true')
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--seed', type=int)
        return parser

    def configure_optimizers(self):
        opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        opt_critic = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.lr)
        return opt_critic, opt_actor


if __name__ == '__main__':
    # Parse args separately so we don't have to abuse **kwargs
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--env', type=str)
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

        env = gym.make(args.env)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_reward_mean',
            dirpath=args.env+'_td3_mlp',
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
        model.eval()

        # Warning: PyBullet environments are hardcoded to record at 320x240
        # There seems to be no easy way to deal with this
        rewards, lengths = model.evaluate(
            num_eval_episodes=10,
            render=True,
            record=True,
            record_fn=args.video_fn)
        print('Mean rewards and length:', np.mean(rewards), np.mean(lengths))
