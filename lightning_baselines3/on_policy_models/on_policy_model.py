import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from contextlib import ExitStack

import gym
import numpy as np

import torch
import pytorch_lightning as pl

from lightning_baselines3.common.base_model import BaseModel
from lightning_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from lightning_baselines3.common.type_aliases import GymEnv, GymObs
from lightning_baselines3.common.vec_env import VecEnv



class OnPolicyModel(BaseModel):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param buffer_length: (int) Length of the buffer and the number of steps to run for each environment per update
    :param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
        just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
    :param batch_size: Minibatch size for each gradient update
    :param epochs_per_rollout: Number of epochs to optimise the loss for
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param seed: Seed for the pseudo random generators
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        buffer_length: int,
        num_rollouts: int,
        batch_size: int,
        epochs_per_rollout: int,
        num_eval_episodes: int = 100,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        super(OnPolicyModel, self).__init__(
            env=env,
            eval_env=eval_env,
            num_eval_episodes=num_eval_episodes,
            verbose=verbose,
            support_multi_env=True,
            seed=seed,
            use_sde=use_sde,
        )

        self.buffer_length = buffer_length
        self.num_rollouts = num_rollouts
        self.batch_size = batch_size
        self.epochs_per_rollout = epochs_per_rollout
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.rollout_buffer = RolloutBuffer(
            buffer_length,
            self.observation_space,
            self.action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.save_hyperparameters()


    def forward(self, obs: GymObs) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Override this function with the forward function of your model
        :param obs: The input observations
        :return: The chosen actions
        """
        raise NotImplementedError


    def train_dataloader(self):
        """
        Create the dataloader for our OffPolicyModel
        """
        return OnPolicyDataloader(self)


    def collect_rollouts(self) -> RolloutBufferSamples:
        """
        Collect rollouts and put them into the RolloutBuffer
        """
        assert self._last_obs is not None, "No previous observation was provided"
        with torch.no_grad():
            # Sample new weights for the state dependent exploration
            if self.use_sde:
                self.reset_noise(self.env.num_envs)

            self.eval()
            for i in range(self.buffer_length):
                if self.use_sde and self.sde_sample_freq > 0 and i % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.reset_noise(self.env.num_envs)

                # Convert to pytorch tensor, let Lightning take care of any GPU transfer
                obs_tensor = torch.as_tensor(self._last_obs).to(device=self.device, dtype=torch.float32)
                dist, values = self(obs_tensor)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions
                # Clip the actions to avoid out of bound error
                if isinstance(self.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
                elif isinstance(self.action_space, gym.spaces.Discrete):
                    clipped_actions = actions.astype(np.int32)

                new_obs, rewards, dones, infos = self.env.step(clipped_actions)

                # Give access to local variables

                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)
                self.rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
                self._last_obs = new_obs
                self._last_dones = dones

            final_obs = torch.as_tensor(new_obs).to(device=self.device, dtype=torch.float32)
            dist, final_values = self(final_obs)
            samples = self.rollout_buffer.finalize(final_values, dones)

            self.rollout_buffer.reset()
        self.train()
        return samples



class OnPolicyDataloader:
    def __init__(self, model: OnPolicyModel):
        self.model = model


    def __iter__(self):
        for i in range(self.model.num_rollouts):
            experiences = self.model.collect_rollouts()
            observations, actions, old_values, old_log_probs, advantages, returns = experiences
            for j in range(self.model.epochs_per_rollout):
                k = 0
                perm = torch.randperm(observations.shape[0], device=observations.device)
                while k < observations.shape[0]:
                    batch_size = min(observations.shape[0] - k, self.model.batch_size)
                    yield RolloutBufferSamples(
                        observations[perm[k:k+batch_size]],
                        actions[perm[k:k+batch_size]],
                        old_values[perm[k:k+batch_size]],
                        old_log_probs[perm[k:k+batch_size]],
                        advantages[perm[k:k+batch_size]],
                        returns[perm[k:k+batch_size]])
                    k += batch_size
