import warnings
from typing import Generator, NamedTuple, Optional, Union

import numpy as np
import torch
from gym import spaces

from lightning_baselines3.common.utils import get_action_dim, get_obs_shape



class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor



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

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_envs: int = 1,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, n_envs=n_envs)

        self.batch_size = batch_size

        assert n_envs == 1, "Replay buffer only support single environment for now"

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)


    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray) -> None:
        self.observations[self.pos] = np.array(obs).copy()
        # A little redundant, but makes the code a lot simpler
        self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


    def sample(self) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=self.batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=self.batch_size)

        next_obs = self.observations[(batch_inds + 1) % self.buffer_size, 0, :]

        # Disable normalisation for now, it would break something like cartpole
        #if self.full:
            #rewards = (self.rewards[batch_inds] - np.mean(self.rewards)) / (np.std(self.rewards) + 1e-6)
        #else:
            #rewards = (self.rewards[batch_inds] - np.mean(self.rewards[:self.pos])) / (np.std(self.rewards[:self.pos]) + 1e-6)
        rewards = self.rewards[batch_inds]

        return ReplayBufferSamples(
            torch.as_tensor(self.observations[batch_inds, 0, :]),
            torch.as_tensor(self.actions[batch_inds, 0, :]),
            torch.as_tensor(next_obs),
            torch.as_tensor(self.dones[batch_inds]),
            torch.as_tensor(rewards))



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
        self.initialised = False

        super(RolloutBuffer, self).reset()

    def finalize(self, last_values: torch.Tensor, last_dones: torch.Tensor) -> RolloutBufferSamples:
        """
        Finalize and compute the returns (sum of discounted rewards) and GAE advantage.
        Adapted from Stable-Baselines PPO2.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        :param last_values: (torch.Tensor) estimated value of the current state
            following the current policy.
        :param last_dones: (torch.Tensor) End of episode signal.
        """
        assert self.full, "Can only finalize RolloutBuffer when RolloutBuffer is full"

        assert last_values.device == self.values.device, 'All value function outputs must be on same device'

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
        self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, value: torch.Tensor, log_prob: torch.Tensor
    ) -> None:
        """
        :param obs: (torch.tensor) Observation
        :param action: (torch.tensor) Action
        :param reward: (torch.tensor)
        :param done: (torch.tensor) End of episode signal.
        :param value: (torch.Tensor) estimated value of the current state
            following the current policy.
        :param log_prob: (torch.Tensor) log probability of the action
            following the current policy.
        """

        # Initialise the first time we add something, so we know which device to put things on
        if not self.initialised:
            self.observations = torch.zeros((self.buffer_size, self.n_envs) + self.obs_shape, device=obs.device)
            self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), device=action.device)
            self.rewards = torch.zeros((self.buffer_size, self.n_envs), device=reward.device)
            self.dones = torch.zeros((self.buffer_size, self.n_envs), device=done.device)
            self.values = torch.zeros((self.buffer_size, self.n_envs), device=value.device)
            self.log_probs = torch.zeros((self.buffer_size, self.n_envs), device=log_prob.device)
            self.initialised = True

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
