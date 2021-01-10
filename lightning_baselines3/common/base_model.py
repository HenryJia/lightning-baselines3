"""Abstract base classes for RL algorithms."""

import io
import inspect
import pathlib
import time
import copy
import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

from lightning_baselines3.common.monitor import Monitor
from lightning_baselines3.common.type_aliases import GymEnv
from lightning_baselines3.common.utils import (
    check_for_correct_spaces,
    get_schedule_fn,
    set_random_seed,
    update_learning_rate,
)
from lightning_baselines3.common.vec_env import VecEnv
from lightning_baselines3.common.vec_env import is_wrapped, wrap_env


def maybe_make_env(env: Union[GymEnv, str], monitor_wrapper: bool, verbose: int) -> Optional[GymEnv]:
    """If env is a string, make the environment; otherwise, return env.

    :param env: (Union[GymEnv, str, None]) The environment to learn from.
    :param monitor_wrapper: (bool) Whether to wrap env in a Monitor when creating env.
    :param verbose: (int) logging verbosity
    :return A Gym (vector) environment.
    """
    if isinstance(env, str):
        if verbose >= 1:
            print(f"Creating environment from the given name '{env}'")
        env = gym.make(env)
        if monitor_wrapper:
            env = Monitor(env, filename=None)

    return env



class BaseModel(pl.LightningModule):
    """
    The base of RL algorithms

    :param env: (Union[GymEnv, str, None]) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param learning_rate: (float or callable) learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param verbose: (int) The verbosity level: 0 none, 1 training information, 2 debug
    :param support_multi_env: (bool) Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param monitor_wrapper: (bool) When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: (Optional[int]) Seed for the pseudo random generators
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        num_eval_episodes: int = 100,
        verbose: int = 0,
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
    ):
        super().__init__()

        self.num_eval_episodes = num_eval_episodes
        self.verbose = verbose

        # When using VecNormalize:
        self._episode_num = 0
        # Used for gSDE only
        self.use_sde = use_sde

        # Create the env for training and evaluation
        self.env = maybe_make_env(env, monitor_wrapper, self.verbose)
        self.eval_env = maybe_make_env(eval_env, monitor_wrapper, self.verbose)

        # Wrap the env if necessary
        self.env = wrap_env(self.env, self.verbose)
        self.eval_env = wrap_env(self.eval_env, self.verbose)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.n_envs = self.env.num_envs

        if seed:
            self.seed = seed
            self.set_random_seed(self.seed)

        if not support_multi_env and self.n_envs > 1:
            raise ValueError(
                "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
            )

        if self.use_sde and not isinstance(self.observation_space, gym.spaces.Box):
            raise ValueError("generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

        self.reset()


    def save_hyperparameters(self, frame=None, exclude=['env', 'eval_env']):
        if not frame:
            frame = inspect.currentframe().f_back
        if not exclude:
            return super().save_hyperparameters(frame=frame)
        if isinstance(exclude, str):
            exclude = (exclude, )
        init_args = pl.utilities.parsing.get_init_args(frame)
        include = [k for k in init_args.keys() if k not in exclude]
        return super().save_hyperparameters(*include, frame=frame)


    def evaluate(
        self,
        num_eval_episodes: int,
        deterministic: bool = True,
        render: bool = False,
        record: bool = False,
        record_fn: Optional[str] = None) -> Tuple[List[float], List[int]]:
        """
        Evaluate the model with eval_env

        :param deterministic: (bool) Whether to evaluate deterministically
        :return: (torch.Tensor) The predicted action and state
        """
        if isinstance(self.eval_env, VecEnv):
            assert self.eval_env.num_envs == 1, "Cannot run eval_env in parallel. eval_env.num_env must equal 1"

        if not is_wrapped(self.eval_env, Monitor) and self.verbose:
            warnings.warn(
                "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
                "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
                "Consider wrapping environment first with ``Monitor`` wrapper.",
                UserWarning,
            )

        episode_rewards, episode_lengths = [], []

        if record:
            assert render, "Cannot record without rendering"
            recorder = VideoRecorder(env=self.eval_env, path=record_fn)

        not_reseted = True
        for i in range(num_eval_episodes):
            done = False
            episode_rewards += [0.0]
            episode_lengths += [0]

            # Number of loops here might differ from true episodes
            # played, if underlying wrappers modify episode lengths.
            # Avoid double reset, as VecEnv are reset automatically.
            if not isinstance(self.eval_env, VecEnv) or not_reseted:
                obs = self.eval_env.reset()
                not_reseted = False

            while not done:
                with torch.no_grad():
                    obs = torch.tensor(obs).to(self.device)
                    action = self.predict(obs, deterministic=deterministic)

                if isinstance(self.action_space, gym.spaces.Box):
                    action = np.clip(action, self.action_space.low, self.action_space.high)
                elif isinstance(self.action_space, gym.spaces.Discrete):
                    action = action.astype(np.int32)

                obs, reward, done, info = self.eval_env.step(action)
                episode_rewards[-1] += reward
                episode_lengths[-1] += 1

                if render:
                    self.eval_env.render()
                    if record:
                        recorder.capture_frame()

            if is_wrapped(self.eval_env, Monitor):
                # Do not trust "done" with episode endings.
                # Remove vecenv stacking (if any)
                if isinstance(self.eval_env, VecEnv):
                    info = info[0]
                if "episode" in info.keys():
                    # Monitor wrapper includes "episode" key in info if environment
                    # has been wrapped with it. Use those rewards instead.
                    episode_rewards[-1] = info["episode"]["r"]
                    episode_lengths[-1] = info["episode"]["l"]
        if record:
            recorder.close()

        return episode_rewards, episode_lengths


    def training_epoch_end(self, outputs):
        """" Run the evaluation function """
        rewards, lengths = self.evaluate(self.num_eval_episodes)
        self.log_dict({
            'val_reward_mean': np.mean(rewards),
            'val_reward_std': np.std(rewards),
            'val_lengths_mean': np.mean(lengths),
            'val_lengths_std': np.std(lengths)},
            prog_bar=True, logger=True)


    def reset(self) -> None: # Reset the environment
        self._last_obs = self.env.reset()
        self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)  # type: Optional[np.ndarray]


    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed: (int)
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == torch.device("cuda").type)
        self.action_space.seed(seed)
        if self.env:
            self.env.seed(seed)
        if self.eval_env:
            self.eval_env.seed(seed)
