from typing import Any, Dict, Optional, Type, Union

import numpy as np
from gym import spaces

import torch
import torch.nn.functional as F
from torch import distributions

from lightning_baselines3.on_policy_models.on_policy_model import OnPolicyModel
from lightning_baselines3.common.type_aliases import GymEnv
from lightning_baselines3.common.utils import explained_variance



class A2C(OnPolicyModel):
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param buffer_length: (int) Length of the buffer and the number of steps to run for each environment per update
    :param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
        just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
    :param batch_size: Minibatch size for each gradient update
    :param epochs_per_rollout: Number of epochs to optimise the loss for
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param value_coef: Value function coefficient for the loss calculation
    :param entropy_coef: Entropy coefficient for the loss calculation
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param seed: Seed for the pseudo random generators
    """
    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        buffer_length: int = 5,
        num_rollouts: int = 100,
        batch_size: int = 128,
        epochs_per_rollout: int = 1,
        num_eval_episodes: int = 100,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        super(A2C, self).__init__(
            env=env,
            eval_env=eval_env,
            buffer_length=buffer_length,
            num_rollouts=num_rollouts,
            batch_size=batch_size,
            epochs_per_rollout=epochs_per_rollout,
            num_eval_episodes=num_eval_episodes,
            gamma=gamma,
            gae_lambda=gae_lambda,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            verbose=verbose,
            seed=seed
        )

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        """
        Specifies the update step for A2C. Override this if you wish to modify the A2C algorithm
        """
        if self.use_sde:
            self.reset_noise(self.batch_size)

        dist, values = self(batch.observations)
        log_probs = dist.log_prob(batch.actions)
        values = values.flatten()

        advantages = batch.advantages.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(advantages * log_probs).mean()
        value_loss = F.mse_loss(batch.returns.detach(), values)
        entropy_loss = -dist.entropy().mean()

        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        with torch.no_grad():
            explained_var = explained_variance(batch.old_values, batch.returns)
        self.log_dict({
            'train_loss': loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'explained_var': explained_var},
            prog_bar=False, logger=True)

        return loss
