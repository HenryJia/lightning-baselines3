from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

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

    :param env: The environment to learn from
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param batch_size: Minibatch size for each gradient update
    :param buffer_length: length of the replay buffer
    :param warmup_length: how many steps of the model to collect transitions for before learning starts
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param episodes_per_rollout: Update the model every ``episodes_per_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
        just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
    :param gradient_steps: How many gradient steps to do after each rollout
    :param target_update_interval: How many environment steps to wait between updating the target Q network
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param gamma: the discount factor
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug (default: 0)
    :param seed: Seed for the pseudo random generators
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
            seed=seed,
            use_sde=False, # DQN Does not support SDE since DQN only supports Discrete actions spaces
            use_sde_at_warmup=False,
            sde_sample_freq=-1
        )

        assert isinstance(self.action_space, spaces.Discrete), "DQN only supports environments with Discrete and Box action spaces"

        self.target_update_interval = target_update_interval


    def reset(self):
        """
        Resets the environment and the counter to keep track of target network updates
        """
        super(DQN, self).reset()
        self.update_timestep = 0


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the Q network.
        Override this function with your own.

        :param x: The input observations
        :return: The output Q values of the Q network
        """
        raise NotImplementedError


    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the target Q network.
        Override this function with your own.

        :param x: The input observations
        :return: The output Q values of the target Q network
        """
        raise NotImplementedError


    def update_target(self):
        """
        Function to update the target Q network periodically.
        Override this function with your own.
        """
        raise NotImplementedError


    def training_step(self, batch, batch_idx):
        """
        Specifies the update step for DQN. Override this if you wish to modify the A2C algorithm
        """
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
