import glob
import os
import random
from collections import deque
from typing import Callable, Iterable, Optional, Tuple, Union

import gym
from gym import spaces
import numpy as np

import torch
import torch.nn.functional as F

from lightning_baselines3.common.type_aliases import GymEnv
from lightning_baselines3.common.vec_env import is_image_space
from lightning_baselines3.common.vec_env import VecTransposeImage

# Check if tensorboard is available for pytorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def preprocess_obs(obs: torch.Tensor, observation_space: spaces.Space, normalize_images: bool = True) -> torch.Tensor:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.

    :param obs: (torch.Tensor) Observation
    :param observation_space: (spaces.Space)
    :param normalize_images: (bool) Whether to normalize images or not
        (True by default)
    :return: (torch.Tensor)
    """
    if isinstance(observation_space, spaces.Box):
        if is_image_space(observation_space) and normalize_images:
            return obs.float() / 255.0
        return obs.float()

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return torch.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(torch.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()

    else:
        raise NotImplementedError()


def get_obs_shape(observation_space: spaces.Space) -> Tuple[int, ...]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space: (spaces.Space)
    :return: (Tuple[int, ...])
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    else:
        raise NotImplementedError()


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    :param observation_space: (spaces.Space)
    :return: (int)
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space: (spaces.Space)
    :return: (int)
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError()


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators
    :param seed: (int)
    :param using_cuda: (bool)
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# From stable baselines
def explained_variance(y_pred: torch.tensor, y_true: torch.tensor) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = torch.var(y_true)
    return torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y


def update_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer: (torch.optim.Optimizer)
    :param learning_rate: (float)
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def get_schedule_fn(value_schedule: Union[Callable, float]) -> Callable:
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


def get_linear_fn(start: float, end: float, end_fraction: float) -> Callable:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: (float) value to start with if ``progress_remaining`` = 1
    :params end: (float) value to end with if ``progress_remaining`` = 0
    :params end_fraction: (float) fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return: (Callable)
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_fn(val: float) -> Callable:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (Callable)
    """

    def func(_):
        return val

    return func


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: (Union[str, torch.device]) One for 'auto', 'cuda', 'cpu'
    :return: (torch.device)
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def get_latest_run_id(log_path: Optional[str] = None, log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :return: (int) latest run number
    """
    max_run_id = 0
    for path in glob.glob(f"{log_path}/{log_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def check_for_correct_spaces(env: GymEnv, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
    """
    Checks that the environment has same spaces as provided ones. Used by BaseAlgorithm to check if
    spaces match after loading the model with given env.
    Checked parameters:
    - observation_space
    - action_space

    :param env: (GymEnv) Environment to check for valid spaces
    :param observation_space: (gym.spaces.Space) Observation space to check against
    :param action_space: (gym.spaces.Space) Action space to check against
    """
    if (
        observation_space != env.observation_space
        # Special cases for images that need to be transposed
        and not (
            is_image_space(env.observation_space)
            and observation_space == VecTransposeImage.transpose_space(env.observation_space)
        )
    ):
        raise ValueError(f"Observation spaces do not match: {observation_space} != {env.observation_space}")
    if action_space != env.action_space:
        raise ValueError(f"Action spaces do not match: {action_space} != {env.action_space}")


def is_vectorized_observation(observation: np.ndarray, observation_space: gym.spaces.Space) -> bool:
    """
    For every observation type, detects and validates the shape,
    then returns whether or not the observation is vectorized.

    :param observation: (np.ndarray) the input observation to validate
    :param observation_space: (gym.spaces) the observation space
    :return: (bool) whether the given observation is vectorized or not
    """
    if isinstance(observation_space, gym.spaces.Box):
        if observation.shape == observation_space.shape:
            return False
        elif observation.shape[1:] == observation_space.shape:
            return True
        else:
            raise ValueError(
                f"Error: Unexpected observation shape {observation.shape} for "
                + f"Box environment, please use {observation_space.shape} "
                + "or (n_env, {}) for the observation shape.".format(", ".join(map(str, observation_space.shape)))
            )
    elif isinstance(observation_space, gym.spaces.Discrete):
        if observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
            return False
        elif len(observation.shape) == 1:
            return True
        else:
            raise ValueError(
                f"Error: Unexpected observation shape {observation.shape} for "
                + "Discrete environment, please use (1,) or (n_env, 1) for the observation shape."
            )

    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        if observation.shape == (len(observation_space.nvec),):
            return False
        elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
            return True
        else:
            raise ValueError(
                f"Error: Unexpected observation shape {observation.shape} for MultiDiscrete "
                + f"environment, please use ({len(observation_space.nvec)},) or "
                + f"(n_env, {len(observation_space.nvec)}) for the observation shape."
            )
    elif isinstance(observation_space, gym.spaces.MultiBinary):
        if observation.shape == (observation_space.n,):
            return False
        elif len(observation.shape) == 2 and observation.shape[1] == observation_space.n:
            return True
        else:
            raise ValueError(
                f"Error: Unexpected observation shape {observation.shape} for MultiBinary "
                + f"environment, please use ({observation_space.n},) or "
                + f"(n_env, {observation_space.n}) for the observation shape."
            )
    else:
        raise ValueError(
            "Error: Cannot determine if the observation is vectorized " + f" with the space type {observation_space}."
        )


def safe_mean(arr: Union[np.ndarray, list, deque]) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr:
    :return:
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def polyak_update(params: Iterable[torch.nn.Parameter], target_params: Iterable[torch.nn.Parameter], tau: float) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the th.main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    :param params: (Iterable[torch.nn.Parameter]) parameters to use to update the target params
    :param target_params: (Iterable[torch.nn.Parameter]) parameters to update
    :param tau: (float) the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
