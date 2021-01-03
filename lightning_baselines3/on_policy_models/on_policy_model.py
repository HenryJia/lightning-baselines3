import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from contextlib import ExitStack

import gym
import numpy as np

import torch
import pytorch_lightning as pl

from lightning_baselines3.common.base_model import BaseModel
from lightning_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from lightning_baselines3.common.type_aliases import GymEnv
from lightning_baselines3.common.utils import safe_mean
from lightning_baselines3.common.vec_env import VecEnv



class OnPolicyModel(BaseModel):
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
    :param sde_sample_freq: (bool) Whether to store gradients in the RolloutBuffer
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
        buffer_length: int,
        num_rollouts: int,
        batch_size: int,
        epochs_per_rollout: int,
        num_eval_episodes: int = 100,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        verbose: int = 0,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
    ):
        super(OnPolicyModel, self).__init__(
            env=env,
            eval_env=eval_env,
            num_eval_episodes=num_eval_episodes,
            verbose=verbose,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
        )

        self.buffer_length = buffer_length
        self.num_rollouts = num_rollouts
        self.batch_size = batch_size
        self.epochs_per_rollout = epochs_per_rollout
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.rollout_buffer = RolloutBuffer(
            buffer_length,
            self.observation_space,
            self.action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )


    def train_dataloader(self):
        return OnPolicyDataloader(self)


    def collect_rollouts(self) -> RolloutBufferSamples:
        assert self._last_obs is not None, "No previous observation was provided"
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.reset_noise(self.env.num_envs)

        for i in range(self.buffer_length):
            if self.use_sde and self.sde_sample_freq > 0 and i % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.reset_noise(self.env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor, let Lightning take care of any GPU transfer
                obs_tensor = torch.as_tensor(self._last_obs).to(device=self.device, dtype=torch.float32)
                dist, values = self(obs_tensor)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            elif isinstance(self.action_space, gym.spaces.Discrete):
                clipped_actions = actions.astype(np.int32)

            new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            # Give access to local variables

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            self.rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        samples = self.rollout_buffer.finalize(values, dones)
        self.rollout_buffer.reset()
        return samples



class OnPolicyDataloader:
    def __init__(self, model: OnPolicyModel):
        self.model = model


    def __iter__(self):
        for i in range(self.model.num_rollouts):
            experiences = self.model.collect_rollouts()
            observations, actions, old_values, old_log_probs, advantages, returns = experiences
            for j in range(self.model.epochs_per_rollout):
                k = 0
                perm = torch.randperm(observations.shape[0], device=observations.device)
                while k < observations.shape[0]:
                    batch_size = min(observations.shape[0] - k, self.model.batch_size)
                    yield RolloutBufferSamples(
                        observations[perm[k:k+batch_size]],
                        actions[perm[k:k+batch_size]],
                        old_values[perm[k:k+batch_size]],
                        old_log_probs[perm[k:k+batch_size]],
                        advantages[perm[k:k+batch_size]],
                        returns[perm[k:k+batch_size]])
                    k += batch_size
