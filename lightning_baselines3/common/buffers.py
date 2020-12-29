import warnings
from typing import Generator, NamedTuple, Optional, Union

import numpy as np
import torch
from gym import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from lightning_baselines3.common.utils import get_action_dim, get_obs_shape
from lightning_baselines3.common.type_aliases import ReplayBufferSamples
from lightning_baselines3.common.vec_env import VecNormalize



class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor



class BaseBuffer(object):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[torch.device, str]) PyTorch device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr: (np.ndarray)
        :return: (np.ndarray)
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        if isinstance(arr, torch.Tensor): # If we're working with torch.Tensor, use view to avoid a copy
            return torch.transpose(arr, 0, 1).view(shape[0] * shape[1], *shape[2:])
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    #@staticmethod
    #def to_torch(
        #array: Union[np.ndarray, torch.Tensor], copy: bool = False, device: Union[torch.Device, None] = None
        #) -> torch.Tensor:
        #"""
        #Convert a numpy array to a PyTorch tensor and copy it to a device
        #Note: It does not copy by default

        #:param array: (np.ndarray)
        #:param copy: (bool) Whether to copy or not the data
            #(may be useful to avoid changing things be reference)
        #:param device: (torch.Device, None) Which device to copy the tensor to, set to None for no copy
        #:return: (torch.Tensor)
        #"""
        #if not if isinstance(array, torch.Tensor):
            #if copy:
                #array = torch.tensor(array)
            #else:
                #array = torch.as_tensor(array)
        #if device:
            #array = array.pin_memory().to(device)
        #return array

    def size(self) -> int:
        """
        :return: (int) The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the bufferrollout_buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:

        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    #def sample(self, env: Optional[VecNormalize] = None):
        #"""
        #:param env: (Optional[VecNormalize]) associated gym VecEnv
            #to normalize the observations/rewards when sampling
        #:return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        #"""
        #upper_bound = self.buffer_size if self.full else self.pos
        #batch_inds = np.random.randint(0, upper_bound, size=self.batch_size)
        #return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None):
        """
        :param batch_inds: (torch.Tensor)
        :param env: (Optional[VecNormalize])
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        raise NotImplementedError()

    @staticmethod
    def _normalize_obs(obs: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_obs(obs).astype(np.float32)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


#class ReplayBuffer(BaseBuffer):
    #"""
    #Replay buffer used in off-policy algorithms like SAC/TD3.

    #:param buffer_size: (int) Max number of element in the buffer
    #:param observation_space: (spaces.Space) Observation space
    #:param action_space: (spaces.Space) Action space
    #:param device: (torch.device)
    #:param n_envs: (int) Number of parallel environments
    #:param optimize_memory_usage: (bool) Enable a memory efficient variant
        #of the replay buffer which reduces by almost a factor two the memory used,
        #at a cost of more complexity.
        #See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        #and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    #"""

    #def __init__(
        #self,
        #buffer_size: int,
        #batch_size: int,
        #observation_space: spaces.Space,
        #action_space: spaces.Space,
        #device: Union[torch.device, str] = "cpu",
        #n_envs: int = 1,
        #optimize_memory_usage: bool = False,
    #):
        #super(ReplayBuffer, self).__init__(buffer_size, batch_size, observation_space, action_space, device, n_envs=n_envs)

        #assert n_envs == 1, "Replay buffer only support single environment for now"

        ## Check that the replay buffer can fit into the memory
        #if psutil is not None:
            #mem_available = psutil.virtual_memory().available

        #self.optimize_memory_usage = optimize_memory_usage
        #self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        #if optimize_memory_usage:
            ## `observations` contains also the next observation
            #self.next_observations = None
        #else:
            #self.next_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        #self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        #self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        #self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        #if psutil is not None:
            #total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            #if self.next_observations is not None:
                #total_memory_usage += self.next_observations.nbytes

            #if total_memory_usage > mem_available:
                ## Convert to GB
                #total_memory_usage /= 1e9
                #mem_available /= 1e9
                #warnings.warn(
                    #"This system does not have apparently enough memory to store the complete "
                    #f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                #)

    #def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray) -> None:
        ## Copy to avoid modification by reference
        #self.observations[self.pos] = np.array(obs).copy()
        #if self.optimize_memory_usage:
            #self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        #else:
            #self.next_observations[self.pos] = np.array(next_obs).copy()

        #self.actions[self.pos] = np.array(action).copy()
        #self.rewards[self.pos] = np.array(reward).copy()
        #self.dones[self.pos] = np.array(done).copy()

        #self.pos += 1
        #if self.pos == self.buffer_size:
            #self.full = True
            #self.pos = 0

    #def sample(self, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        #"""
        #Sample elements from the replay buffer.
        #Custom sampling when using memory efficient variant,
        #as we should not sample the element with index `self.pos`
        #See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        #:param env: (Optional[VecNormalize]) associated gym VecEnv
            #to normalize the observations/rewards when sampling
        #:return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        #"""
        #if not self.optimize_memory_usage:
            #return super().sample(batch_size=self.batch_size, env=env)
        ## Do not sample the element with index `self.pos` as the transitions is train_loader = DataLoader(MNIST(os.getcwd(), download=True, transform=transforms.ToTensor()))invalid
        ## (we use only one array to store `obs` and `next_obs`)
        #if self.full:
            #batch_inds = (np.random.randint(1, self.buffer_size, size=self.batch_size) + self.pos) % self.buffer_size
        #else:
            #batch_inds = np.random.randint(0, self.pos, size=self.batch_size)
        #return self._get_samples(batch_inds, env=env)

    #def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        #if self.optimize_memory_usage:
            #next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        #else:
            #next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        #data = (
            #self._normalize_obs(self.observations[batch_inds, 0, :], env),
            #self.actions[batch_inds, 0, :],
            #next_obs,
            #self.dones[batch_inds],
            #self._normalize_reward(self.rewards[batch_inds], env),
        #)
        #return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (torch.device)
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: (float) Discount factor
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        gamma: float = 0.99,
        gae_lambda: float = 1,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.reset()

    def reset(self) -> None:

        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

        super(RolloutBuffer, self).reset()

    def finalize(self, last_values: torch.Tensor, last_dones: np.ndarray) -> RolloutBufferSamples:
        """
        Finalize and compute the returns (sum of discounted rewards) and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param dones: (np.ndarray)

        """
        assert self.full, "Can only finalize RolloutBuffer when RolloutBuffer is full"

        self.observations = np.stack(self.observations, axis=0)
        self.actions = np.stack(self.actions, axis=0)
        self.rewards = np.stack(self.rewards, axis=0)
        self.dones = np.stack(self.dones, axis=0)
        self.values = torch.stack(self.values, dim=0)
        self.log_probs = torch.stack(self.log_probs, dim=0)

        assert last_values.device == self.values.device, 'All value function outputs must be on same device'

        # Move everything to torch
        # Lightning can handle moving things to device to some extent, but we need to make sure everything
        # is consistent for computing advantages and returns
        self.observations = torch.as_tensor(self.observations).float()
        self.actions = torch.as_tensor(self.actions).float()
        self.rewards = torch.as_tensor(self.rewards).to(device=last_values.device, dtype=torch.float32)
        self.dones = torch.as_tensor(self.dones).to(device=last_values.device, dtype=torch.int32)
        last_dones = torch.as_tensor(last_dones).to(device=last_values.device, dtype=torch.int32)

        last_gae_lam = 0
        advantages = torch.zeros_like(self.rewards)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        returns = advantages + self.values

        self.observations = self.observations.view((-1, *self.observations.shape[2:]))
        self.actions = self.actions.view((-1, *self.actions.shape[2:]))
        self.rewards = self.rewards.flatten()
        self.values = self.values.flatten()
        self.log_probs = self.log_probs.flatten()
        advantages = advantages.flatten()
        returns = returns.flatten()

        return RolloutBufferSamples(self.observations, self.actions, self.values, self.log_probs, advantages, returns)


    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, value: torch.Tensor, log_prob: torch.Tensor
    ) -> None:
        """
        :param obs: (np.ndarray) Observation
        :param action: (np.ndarray) Action
        :param reward: (np.ndarray)
        :param done: (np.ndarray) End of episode signal.
        :param value: (torch.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (torch.Tensor) log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob[:, None]

        self.observations += [np.array(obs)]
        self.actions += [np.array(action)]
        self.rewards += [np.array(reward)]
        self.dones += [np.array(done)]
        self.values += [value.flatten()]
        self.log_probs += [log_prob]
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
