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
    The base for Off-Policy algorithms (ex: SAC/TD3)

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
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param gamma: the discount factor
    :param squashed_actions: whether the actions are squashed between [-1, 1] and need to be unsquashed
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
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
        train_freq: int = -1,
        episodes_per_rollout: int = -1,
        num_rollouts: int = 1,
        gradient_steps: int = 1,
        num_eval_episodes: int = 10,
        gamma: float = 0.99,
        squashed_actions: bool = False,
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

        assert self.env.num_envs == 1, "OffPolicyModel only support single environment at this stage"
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
        self.squashed_actions = squashed_actions

        self.replay_buffer = ReplayBuffer(
            buffer_length,
            batch_size,
            self.observation_space,
            self.action_space,
            n_envs=self.n_envs,
        )


    def reset(self):
        """
        Reset the environment and set the num_timesteps to 0
        """
        super(OffPolicyModel, self).reset()
        self.num_timesteps = 0


    def on_step(self):
        """
        Simple callback for each step we take in the environment
        """
        pass


    def train_dataloader(self):
        """
        Create the dataloader for our OffPolicyModel
        """
        return OffPolicyDataloader(self)


    def scale_actions(
        self, actions: np.ndarray, squashed=False
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale the action appropriately for spaces.Box based on whether they
        are squashed between [-1, 1]

        :param action: The input action
        :return: The action to step the environment with and the action to buffer with
        """

        high, low = self.action_space.high, self.action_space.low
        center = (high + low) / 2.0
        if squashed:
            actions = center + actions * (high - low) / 2.0
        else:
            actions = np.clip(
                actions,
                self.action_space.low,
                self.action_space.high)
        return actions


    def sample_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples an action from the environment or from our model
        :param obs: The input observation
        :param deterministic: Whether we are sampling deterministically.
            This argument has no effect if we are warming up.
        :return: The action to step with, and the action to store in our buffer
        """
        with torch.no_grad():
            # Convert to pytorch tensor
            obs_tensor = torch.as_tensor(obs).to(device=self.device, dtype=torch.float32)
            actions = self.predict(obs_tensor, deterministic=False)

        # Clip and scale actions appropriately
        if isinstance(self.action_space, gym.spaces.Box):
            actions = self.scale_actions(actions, self.squashed_actions)
        elif isinstance(self.action_space, (gym.spaces.Discrete,
                                            gym.spaces.MultiDiscrete,
                                            gym.spaces.MultiBinary)):
            actions = actions.astype(np.int32)
        return actions


    def collect_rollouts(self):
        """
        Collect rollouts and put them into the ReplayBuffer
        """
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
                actions = self.sample_action(self._last_obs, deterministic=False)

            new_obs, rewards, dones, infos = self.env.step(actions)

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

        self.log('num_timesteps', self.num_timesteps, on_step=True, prog_bar=True, logger=True)

        if self.gradient_steps < 1:
            return i
        else:
            return self.gradient_steps

    def training_epoch_end(self, outputs) -> None:
        """
        Run the evaluation function at the end of the training epoch
        Override this if you also wish to do other things at the end of a training epoch
        """
        if self.num_timesteps >= self.warmup_length:
            self.eval()
            rewards, lengths = self.evaluate(self.num_eval_episodes)
            self.train()
            self.log_dict({
                'val_reward_mean': np.mean(rewards),
                'val_reward_std': np.std(rewards),
                'val_lengths_mean': np.mean(lengths),
                'val_lengths_std': np.std(lengths)},
                prog_bar=True, logger=True)


class OffPolicyDataloader:
    def __init__(self, model: OffPolicyModel):
        self.model = model


    def __iter__(self):
        for i in range(self.model.num_rollouts):
            gradient_steps = self.model.collect_rollouts()
            for j in range(gradient_steps):
                yield self.model.replay_buffer.sample()
