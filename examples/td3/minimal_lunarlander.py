import copy

import torch
from torch import nn

import pytorch_lightning as pl

from lightning_baselines3.off_policy_models import TD3
from lightning_baselines3.common.utils import polyak_update


class Model(TD3):
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        # Note: The output layer of the actor must be Tanh activated
        self.actor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.action_space.shape[0]),
            nn.Tanh())

        in_dim = self.observation_space.shape[0] + self.action_space.shape[0]
        self.critic1 = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1))

        self.critic2 = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1))

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target1 = copy.deepcopy(self.critic1)
        self.critic_target2 = copy.deepcopy(self.critic2)

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
        polyak_update(
            self.actor.parameters(),
            self.actor_target.parameters(),
            tau=0.005)
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
        if not deterministic:
            out = out + torch.randn_like(out) * 0.1
        out = torch.clamp(out, -1, 1)
        return out.cpu().numpy()

    def configure_optimizers(self):
        opt_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        opt_critic = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=1e-3)
        return opt_critic, opt_actor


if __name__ == '__main__':
    model = Model(
        env='LunarLanderContinuous-v2',
        eval_env='LunarLanderContinuous-v2',
        warmup_length=10000,
        )

    trainer = pl.Trainer(max_epochs=30, gradient_clip_val=0.5)
    trainer.fit(model)
    print(model.num_timesteps)

    model.evaluate(num_eval_episodes=10, render=True)
