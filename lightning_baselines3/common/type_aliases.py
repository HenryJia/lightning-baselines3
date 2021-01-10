"""Common aliases for type hints"""

from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import gym
import numpy as np
import torch

from lightning_baselines3.common.vec_env import VecEnv

GymEnv = Union[gym.Env, VecEnv]
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]
TensorDict = Dict[str, torch.Tensor]
OptimizerStateDict = Dict[str, Any]
