import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from contextlib import ExitStack

import gym
import numpy as np

import torch
import pytorch_lightning as pl

from lightning_baselines3.common.base_model import BaseModel
from lightning_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from lightning_baselines3.common.type_aliases import GymEnv
from lightning_baselines3.common.vec_env import VecEnv



class OffPolicyModel(BaseModel):
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
        train_freq: int = -1,
        episodes_per_rollout: int = -1,
        num_rollouts: int = 1,
        gradient_steps: int = 1,
        num_eval_episodes: int = 100,
        gamma: float = 0.99,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        super(OffPolicyModel, self).__init__(
            env=env,
            eval_env=eval_env,
            num_eval_episodes=num_eval_episodes,
            verbose=verbose,
            support_multi_env=True,
            seed=seed,
            use_sde=use_sde,
        )

        assert self.env.num_envs == 1, "OffPolicyAlgorithm only support single environment at this stage"
        assert train_freq > 0 or episodes_per_rollout > 0, "At least one of train_freq or episodes_per_rollout must be passed"
        if train_freq > 0 and episodes_per_rollout > 0:
            warnings.warn(
                "You passed a positive value for `train_freq` and `n_episodes_rollout`."
                "Please make sure this is intended. "
                "The agent will collect data by stepping in the environment "
                "until both conditions are true: "
                "`number of steps in the env` >= `train_freq` and "
                "`number of episodes` > `n_episodes_rollout`"
            )

        self.batch_size = batch_size
        self.buffer_length = buffer_length
        self.warmup_length = warmup_length
        self.train_freq = train_freq
        self.episodes_per_rollout = episodes_per_rollout
        self.gradient_steps = gradient_steps
        self.num_rollouts = num_rollouts
        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(
            buffer_length,
            batch_size,
            self.observation_space,
            self.action_space,
            n_envs=self.n_envs,
        )


    def reset(self):
        super(OffPolicyModel, self).reset()
        self.num_timesteps = 0


    def on_step(self):
        """
        Simple callback for each step we take in the environment
        """
        pass


    def train_dataloader(self):
        return OffPolicyDataloader(self)


    def collect_rollouts(self):
        assert self._last_obs is not None, "No previous observation was provided"
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.reset_noise(self.env.num_envs)

        i = 0
        total_episodes = 0

        self.eval()
        while i < self.train_freq or total_episodes < self.episodes_per_rollout:
            if self.use_sde and self.sde_sample_freq > 0 and i % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.reset_noise(self.env.num_envs)

            if self.num_timesteps < self.warmup_length:
                actions = np.array([self.action_space.sample()])
            else:
                with torch.no_grad():
                    # Convert to pytorch tensor, let Lightning take care of any GPU transfer
                    obs_tensor = torch.as_tensor(self._last_obs).to(device=self.device, dtype=torch.float32)
                    actions = self.predict(obs_tensor, deterministic=False)

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            elif isinstance(self.action_space, gym.spaces.Discrete):
                clipped_actions = actions.astype(np.int32)

            new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            self.replay_buffer.add(self._last_obs, new_obs, actions, rewards, dones)

            self._last_obs = new_obs
            i += 1
            self.num_timesteps += 1
            # Note: VecEnv might not return None, it might return [None] or something, remember to double check this!
            if dones:
                total_episodes += 1

            self.on_step()
        self.train()


class OffPolicyDataloader:
    def __init__(self, model: OffPolicyModel):
        self.model = model


    def __iter__(self):
        for i in range(self.model.num_rollouts):
            self.model.collect_rollouts()
            for j in range(self.model.gradient_steps):
                yield self.model.replay_buffer.sample()
