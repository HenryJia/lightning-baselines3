from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pytest

import gym
from gym import spaces

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions

import pytorch_lightning as pl

from lightning_baselines3.off_policy_models.off_policy_model import OffPolicyModel
from lightning_baselines3.common.type_aliases import GymEnv



class DQN(OffPolicyModel):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param buffer_length: (int) Length of the buffer and the number of steps to run for each environment per update
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param sde_sample_freq: (bool) Whether to store gradients in the ReplayBuffer
        Default: False
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param seed: (int) Seed for the pseudo random generators
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        batch_size: int = 256,
        buffer_length: int = int(1e6),
        warmup_length: int = 100,
        train_freq: int = 4,
        episodes_per_rollout: int = -1,
        num_rollouts: int = 1,
        gradient_steps: int = 1,
        target_update_interval: int = 10000,
        num_eval_episodes: int = 100,
        gamma: float = 0.99,
        verbose: int = 0,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
    ):
        super(DQN, self).__init__(
            env=env,
            eval_env=eval_env,
            batch_size=batch_size,
            buffer_length=buffer_length,
            warmup_length=warmup_length,
            train_freq=train_freq,
            episodes_per_rollout=episodes_per_rollout,
            num_rollouts=num_rollouts,
            gradient_steps=gradient_steps,
            num_eval_episodes=num_eval_episodes,
            gamma=gamma,
            verbose=verbose,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=False, # DQN Does not support SDE since DQN only supports Discrete actions spaces
            use_sde_at_warmup=False,
            sde_sample_freq=-1
        )

        assert isinstance(self.action_space, spaces.Discrete), "DQN only supports environments with Discrete and Box action spaces"

        self.target_update_interval = target_update_interval


    def reset(self):
        super(DQN, self).reset()
        self.update_timestep = 0


    def forward_target(self, x):
        """
        Runs the target Q network
        """
        raise NotImplementedError


    def update_target(self):
        """
        Function to update the target Q network periodically
        """
        raise NotImplementedError


    def training_step(self, batch, batch_idx):

        if float(self.num_timesteps - self.update_timestep) / self.target_update_interval > 1:
            self.update_target()
            self.update_timestep = self.num_timesteps

        with torch.no_grad():
            target_q = self.forward_target(batch.next_observations)
            target_q = torch.max(target_q, dim=1, keepdims=True)[0]
            target_q = batch.rewards + (1 - batch.dones) * self.gamma * target_q

        current_q = self(batch.observations)
        current_q = torch.gather(current_q, dim=1, index=batch.actions.long())

        loss = F.smooth_l1_loss(current_q, target_q)
        if self.num_timesteps < self.warmup_length:
            loss = loss * 0
        return loss
