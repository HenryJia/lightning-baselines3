# flake8: noqa F401
import typing
from copy import deepcopy
from typing import Optional, Type, Union

import gym

from lightning_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvWrapper
from lightning_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from lightning_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from lightning_baselines3.common.vec_env.vec_check_nan import VecCheckNan
from lightning_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from lightning_baselines3.common.vec_env.vec_normalize import VecNormalize
from lightning_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from lightning_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from lightning_baselines3.common.vec_env.utils import is_image_space


def wrap_env(env: Union[gym.Env, VecEnv], verbose: int = 0) -> VecEnv:
    if not isinstance(env, VecEnv):
        if verbose >= 1:
            print("Wrapping the env in a DummyVecEnv.")
        env = DummyVecEnv([lambda: env])

    if is_image_space(env.observation_space) and not isinstance(env, VecTransposeImage):
        if verbose >= 1:
            print("Wrapping the env in a VecTransposeImage.")
        env = VecTransposeImage(env)
    return env


def unwrap_wrapper(env: gym.Env, wrapper_class: Type[gym.Wrapper]) -> Optional[gym.Wrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: Type[gym.Env], wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def unwrap_vec_wrapper(env: Union[gym.Env, VecEnv], vec_wrapper_class: Type[VecEnvWrapper]) -> Optional[VecEnvWrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: (gym.Env)
    :param vec_wrapper_class: (VecEnvWrapper)
    :return: (VecEnvWrapper)
    """
    env_tmp = env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, vec_wrapper_class):
            return env_tmp
        env_tmp = env_tmp.venv
    return None


def unwrap_vec_normalize(env: Union[gym.Env, VecEnv]) -> Optional[VecNormalize]:
    """
    :param env: (gym.Env)
    :return: (VecNormalize)
    """
    return unwrap_vec_wrapper(env, VecNormalize)  # pytype:disable=bad-return-type
