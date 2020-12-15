"""Abstract base classes for RL algorithms."""

import io
import pathlib
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

from lightning_baselines3.common.monitor import Monitor
from lightning_baselines3.common.preprocessing import is_image_space
from lightning_baselines3.common.type_aliases import GymEnv
from lightning_baselines3.common.utils import (
    check_for_correct_spaces,
    get_schedule_fn,
    set_random_seed,
    update_learning_rate,
)
from lightning_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize, VecTransposeImage, unwrap_vec_normalize


def maybe_make_env(env: Union[GymEnv, str, None], monitor_wrapper: bool, verbose: int) -> Optional[GymEnv]:
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
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
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
        eval_env: Optional[GymEnv] = None,
        verbose: int = 0,
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
    ):

        if verbose > 0:
            print(f"Using {self.device} device")

        # get VecNormalize object if needed
        self._vec_normalize_env = unwrap_vec_normalize(env)
        self.verbose = verbose
        # Used for updating schedules

        self._last_obs = None  # type: Optional[np.ndarray]
        self._last_dones = None  # type: Optional[np.ndarray]
        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[np.ndarray]
        self._episode_num = 0
        # Used for gSDE only
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq

        # Create the env for training
        self.env = maybe_make_env(env, monitor_wrapper, self.verbose)
        if eval_env: # If we have specificed an evaluation env, use that
            self.eval_env = maybe_make_env(eval_env, monitor_wrapper, self.verbose)
            assert eval_env.num_envs == 1
        else: # Otherwise, use a copy of the training env
            self.eval_env = copy.deepcopy(self.env)

        if seed:
            self.set_random_seed(self.seed)

        # Wrap the env if necessary
        self.env = self._wrap_env(self.env)
        self.eval_env = self._wrap_env(eval_env)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.n_envs = self.env.num_envs

        if not support_multi_env and self.n_envs > 1:
            raise ValueError(
                "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
            )

        if self.use_sde and not isinstance(self.observation_space, gym.spaces.Box):
            raise ValueError("generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")


    def reset(self): # Reset the environment
        self._last_obs = self.env.reset()


    def _wrap_env(self, env: GymEnv) -> VecEnv:
        if not isinstance(env, VecEnv):
            if self.verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])

        if is_image_space(env.observation_space) and not isinstance(env, VecTransposeImage):
            if self.verbose >= 1:
                print("Wrapping the env in a VecTransposeImage.")
            env = VecTransposeImage(env)
        return env


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
