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
from lightning_baselines3.common.type_aliases import GymEnv, GymObs
from lightning_baselines3.common.utils import set_random_seed
from lightning_baselines3.common.vec_env import VecEnv
from lightning_baselines3.common.vec_env import is_wrapped, wrap_env


def maybe_make_env(env: Union[GymEnv, str], verbose: int) -> Optional[GymEnv]:
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

    return env



class BaseModel(pl.LightningModule):
    """
    The base of RL algorithms

    :param env: The environment to learn from
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments in parallel
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        num_eval_episodes: int = 10,
        verbose: int = 0,
        support_multi_env: bool = False,
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
        self.env = maybe_make_env(env, self.verbose)
        self.eval_env = maybe_make_env(eval_env, self.verbose)

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

        if self.use_sde and not isinstance(self.action_space, gym.spaces.Box):
            raise ValueError("generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

        self.reset()


    def predict(self, obs: GymObs, deterministic: bool = False) -> np.ndarray:
        """
        Override this function with the predict function of your own model

        :param obs: The input observations
        :param deterministic: Whether to predict deterministically
        :return: The chosen actions
        """
        raise NotImplementedError


    def save_hyperparameters(self, frame=None, exclude=['env', 'eval_env']):
        """
        Utility function to save the hyperparameters of the model.
        This function behaves identically to LightningModule.save_hyperparameters, but will by default exclude the Gym environments
        See https://pytorch-lightning.readthedocs.io/en/latest/hyperparameters.html#lightningmodule-hyperparameters for more details
        """
        if not frame:
            frame = inspect.currentframe().f_back
        if not exclude:
            return super().save_hyperparameters(frame=frame)
        if isinstance(exclude, str):
            exclude = (exclude, )
        init_args = pl.utilities.parsing.get_init_args(frame)
        include = [k for k in init_args.keys() if k not in exclude]

        if len(include) > 0:
            super().save_hyperparameters(*include, frame=frame)


    def evaluate(
        self,
        num_eval_episodes: int,
        deterministic: bool = True,
        render: bool = False,
        record: bool = False,
        record_fn: Optional[str] = None) -> Tuple[List[float], List[int]]:
        """
        Evaluate the model with eval_env

        :param num_eval_episodes: Number of episodes to evaluate for
        :param deterministic: Whether to evaluate deterministically
        :param render: Whether to render while evaluating
        :param record: Whether to recod while evaluating
        :param record_fn: File to record environment to if we are recording
        :return: A list of total episode rewards and a list of episode lengths
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


    def training_epoch_end(self, outputs) -> None:
        """
        Run the evaluation function at the end of the training epoch
        Override this if you also wish to do other things at the end of a training epoch
        """
        self.eval()
        rewards, lengths = self.evaluate(self.num_eval_episodes)
        self.train()
        self.log_dict({
            'val_reward_mean': np.mean(rewards),
            'val_reward_std': np.std(rewards),
            'val_lengths_mean': np.mean(lengths),
            'val_lengths_std': np.std(lengths)},
            prog_bar=True, logger=True)


    def reset(self) -> None:
        """
        Reset the enviornment
        """
        self._last_obs = self.env.reset()
        self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)


    def set_random_seed(self, seed: int) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym)

        :param seed: The random seed to set
        """
        set_random_seed(seed)
        self.action_space.seed(seed)
        if self.env:
            self.env.seed(seed)
        if self.eval_env:
            self.eval_env.seed(seed)
