from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import math

import gym
from gym import spaces
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions

import pytorch_lightning as pl

from lightning_baselines3.off_policy_models import OffPolicyModel
from lightning_baselines3.common.type_aliases import GymEnv



class SAC(OffPolicyModel):
    """
    Soft Actor Critic (SAC)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param env: The environment to learn from
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param batch_size: Minibatch size for each gradient update
    :param buffer_length: length of the replay buffer
    :param warmup_length: how many steps of the model to collect transitions for before learning starts
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param episodes_per_rollout: Update the model every ``episodes_per_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
        just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
    :param gradient_steps: How many gradient steps to do after each rollout
    :param target_update_interval: How many environment steps to wait between updating the target Q network
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param gamma: the discount factor
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug (default: 0)
    :param seed: Seed for the pseudo random generators
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        batch_size: int = 256,
        buffer_length: int = int(1e6),
        warmup_length: int = 100,
        train_freq: int = 1,
        episodes_per_rollout: int = -1,
        num_rollouts: int = 1000,
        gradient_steps: int = 1,
        target_update_interval: int = 1,
        num_eval_episodes: int = 10,
        gamma: float = 0.99,
        entropy_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        self.entropy_coef = entropy_coef
        self.target_entropy = target_entropy
        self.target_update_interval = target_update_interval

        super(SAC, self).__init__(
            env=env,
            eval_env=eval_env,
            batch_size=batch_size,
            buffer_length=buffer_length,
            warmup_length=warmup_length,
            train_freq=train_freq,
            episodes_per_rollout=episodes_per_rollout,
            num_rollouts=num_rollouts,
            gradient_steps=gradient_steps,
            num_eval_episodes=num_eval_episodes,
            gamma=gamma,
            verbose=verbose,
            seed=seed,
            use_sde=use_sde,
            use_sde_at_warmup=use_sde_at_warmup,
            sde_sample_freq=sde_sample_freq
        )

        assert isinstance(self.action_space, spaces.Box), "SAC only supports environments with Box action spaces"

        # We need manual optimization for this
        self.automatic_optimization = False


    def reset(self):
        """
        Resets the environment and automatic entropy
        """
        super(SAC, self).reset()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.entropy_coef, str):
            if not hasattr(self, 'log_entropy_coef'):
                assert self.entropy_coef.startswith("auto")
                # Default initial value of entropy_coef when learned
                init_value = 1.0
                if "_" in self.entropy_coef:
                    init_value = float(self.entropy_coef.split("_")[1])
                    assert init_value > 0.0, "The initial value of entropy_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                self.log_entropy_coef = torch.log(torch.ones(1, device=self.device) * init_value)
                self.log_entropy_coef = nn.Parameter(self.log_entropy_coef.requires_grad_(True))
                self.entropy_coef_optimizer = torch.optim.Adam([self.log_entropy_coef], lr=3e-4)
        else:
            # I know this isn't very efficient but it makes the code cleaner
            # and it's only one extra operation
            self.log_entropy_coef = torch.log(float(self.entropy_coef))


    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the Q network.
        Override this function with your own.

        :param x: The input observations
        :return: The output Q values of the Q network
        """
        raise NotImplementedError


    def forward_critics(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Runs the target Q network.
        Override this function with your own.

        :param x: The input observations
        :return: The output Q values of the target Q network
        """
        raise NotImplementedError


    def forward_critic_targets(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Runs the target Q network.
        Override this function with your own.

        :param x: The input observations
        :return: The output Q values of the target Q network
        """
        raise NotImplementedError


    def update_targets(self) -> None:
        """
        Function to update the target Q network periodically.
        Override this function with your own.
        """
        raise NotImplementedError


    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Specifies the update step for DQN. Override this if you wish to modify the DQN algorithm
        """
        # We need to sample because `log_std` may have changed between two gradient steps
        if self.num_timesteps < self.warmup_length:
            return

        opt_critic, opt_actor = self.optimizers(use_pl_optimizer=True)

        # Action by the current actor for the sampled state
        dist = self.forward_actor(batch.observations)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        entropy_coef = torch.exp(self.log_entropy_coef)
        if hasattr(self, 'entropy_coef_optimizer'):
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            entropy_coef = entropy_coef.detach()
            entropy_coef_loss = -(self.log_entropy_coef * (log_probs + self.target_entropy).detach()).mean()
            self.log('entropy_coef_loss', entropy_coef_loss, on_step=True, prog_bar=True, logger=True)

        self.log('entropy_coef', entropy_coef, on_step=True, prog_bar=False, logger=True)

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if hasattr(self, 'entropy_coef_optimizer'):
            self.manual_backward(entropy_coef_loss, self.entropy_coef_optimizer)
            self.entropy_coef_optimizer.step()
            self.entropy_coef_optimizer.zero_grad()

        with torch.no_grad():
            # Select action according to policy
            next_dist = self.forward_actor(batch.next_observations)
            next_actions = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions)
            # Compute the target Q value: min over all critics targets
            targets = self.forward_critic_targets(batch.next_observations, next_actions)
            target_q = torch.minimum(*targets)
            # add entropy term
            target_q = target_q - entropy_coef * next_log_probs.reshape(-1, 1)
            # td error + entropy term
            target_q = batch.rewards + (1 - batch.dones) * self.gamma * target_q

        # Get current Q estimates for each critic network
        # using action from the replay buffer
        current_q_estimates = self.forward_critics(batch.observations, batch.actions)

        # Compute critic loss
        critic_loss = torch.stack([F.mse_loss(current_q, target_q) for current_q in current_q_estimates])
        critic_loss = torch.mean(critic_loss)
        self.log('critic_loss', critic_loss, on_step=True, prog_bar=True, logger=True)

        # Optimize the critic
        self.manual_backward(critic_loss, opt_critic)
        opt_critic.step()
        opt_critic.zero_grad()

        # Compute actor loss
        # Alternative: actor_loss = torch.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        q_values_pi = self.forward_critics(batch.observations, actions)
        min_qf_pi = torch.minimum(*q_values_pi).detach()
        actor_loss = (entropy_coef * log_probs - min_qf_pi).mean()
        self.log('actor_loss', actor_loss, on_step=True, prog_bar=True, logger=True)

        # Optimize the actor
        self.manual_backward(actor_loss, opt_actor)
        opt_actor.step()
        opt_actor.zero_grad()

        # Update target networks
        if batch_idx % self.target_update_interval == 0:
            self.update_targets()
