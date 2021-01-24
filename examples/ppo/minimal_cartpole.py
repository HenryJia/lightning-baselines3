import gym

import torch
from torch import distributions
from torch import nn

import pytorch_lightning as pl

from lightning_baselines3.common.vec_env import make_vec_env, SubprocVecEnv
from lightning_baselines3.on_policy_models import PPO


class Model(PPO):
    def __init__(self, **kwargs):
        # **kwargs will pass our arguments on to PPO
        super(Model, self).__init__(**kwargs)

        self.actor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_space.n),
            nn.Softmax(dim=1))

        self.critic = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))

        self.save_hyperparameters()

    # This is for training the model
    # Returns the distribution and the corresponding value
    def forward(self, x):
        out = self.actor(x)
        dist = distributions.Categorical(probs=out)
        return dist, self.critic(x).flatten()

    # This is for inference and evaluation of our model, returns the action
    def predict(self, x, deterministic=True):
        out = self.actor(x)
        if deterministic:
            out = torch.max(out, dim=1)[1]
        else:
            out = distributions.Categorical(probs=out).sample()
        return out.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer


if __name__ == '__main__':
    env = make_vec_env('CartPole-v1', n_envs=4, vec_env_cls=SubprocVecEnv)
    eval_env = gym.make('CartPole-v1')
    model = Model(env=env, eval_env=eval_env)

    trainer = pl.Trainer(max_epochs=5, gradient_clip_val=0.5)
    trainer.fit(model)

    model.evaluate(num_eval_episodes=10, render=True)
