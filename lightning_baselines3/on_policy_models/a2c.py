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
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
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


    def training_step(self, batch, batch_idx):
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
