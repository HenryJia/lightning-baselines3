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


class BaseModel(nn.Module):
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


    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)


    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        """
        Return the ``VecNormalize`` wrapper of the training env
        if it exists.
        :return: Optional[VecNormalize] The ``VecNormalize`` env.
        """
        return self._vec_normalize_env


    def set_env(self, env: GymEnv) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space
        - action_space

        :param env: The environment for learning a policy
        """
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        # it must be coherent now
        # if it is not a VecEnv, make it a VecEnv
        env = self._wrap_env(env)

        self.n_envs = env.num_envs
        self.env = env


    #@abstractmethod
    #def learn(
        #self,
        #total_timesteps: int,
        #callback: MaybeCallback = None,
        #log_interval: int = 100,
        #tb_log_name: str = "run",
        #eval_env: Optional[GymEnv] = None,
        #eval_freq: int = -1,
        #n_eval_episodes: int = 5,
        #eval_log_path: Optional[str] = None,
        #reset_num_timesteps: bool = True,
    #) -> "BaseAlgorithm":
        #"""
        #Return a trained model.

        #:param total_timesteps: (int) The total number of samples (env steps) to train on
        #:param callback: (MaybeCallback) callback(s) called at every step with state of the algorithm.
        #:param log_interval: (int) The number of timesteps before logging.
        #:param tb_log_name: (str) the name of the run for TensorBoard logging
        #:param eval_env: (gym.Env) Environment that will be used to evaluate the agent
        #:param eval_freq: (int) Evaluate the agent every ``eval_freq`` timesteps (this may vary a little)
        #:param n_eval_episodes: (int) Number of episode to evaluate the agent
        #:param eval_log_path: (Optional[str]) Path to a folder where the evaluations will be saved
        #:param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        #:return: (BaseAlgorithm) the trained model
        #"""


    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (Optional[np.ndarray]) The last states (can be None, used in recurrent policies)
        :param mask: (Optional[np.ndarray]) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (Tuple[np.ndarray, Optional[np.ndarray]]) the model's action and the next state
            (used in recurrent policies)
        """


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
