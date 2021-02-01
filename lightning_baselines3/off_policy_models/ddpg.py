from typing import Optional, Tuple, Union

from gym import spaces

import torch
import torch.nn.functional as F

from lightning_baselines3.off_policy_models import TD3
from lightning_baselines3.common.type_aliases import GymEnv


class DDPG(TD3):
    """
    Deep Deterministic Policy Gradient (DDPG).

    Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    Note: we treat DDPG as a special case of its successor TD3.

    :param env: The environment to learn from. If registered in Gym,
        can be str. Can be None for loading trained models
    :param eval_env: The environment to evaluate on, must not be parallelrised.
        If registered in Gym, can be str.
        Can be None for loading trained models
    :param batch_size: Minibatch size for each gradient update
    :param buffer_length: length of the replay buffer
    :param warmup_length: how many steps of the model to collect transitions
        for before learning starts
    :param train_freq: Update the model every ``train_freq`` steps.
        Set to `-1` to disable.
    :param episodes_per_rollout: Update the model every ``episodes_per_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``.
        Set to `-1` to disable.
    :param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch.
        This does not affect any training dynamic, just how often we evaluate the model since
        evaluation happens at the end of each Lightning epoch
    :param gradient_steps: How many gradient steps to do after each rollout
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param squashed_actions: whether the actions are squashed between [-1, 1] and need to be unsquashed
    :param gamma: the discount factor
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug (default: 0)
    :param seed: Seed for the pseudo random generators
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        batch_size: int = 128,
        buffer_length: int = int(1e6),
        warmup_length: int = 100,
        train_freq: int = -1,
        episodes_per_rollout: int = 1,
        num_rollouts: int = 10,
        gradient_steps: int = -1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        num_eval_episodes: int = 10,
        gamma: float = 0.99,
        squashed_actions: bool = True,
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        super(DDPG, self).__init__(
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
            squashed_actions=squashed_actions)

        self.n_critics = 1  # Set this to 1 for DDPG

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Runs the actor network.
        Override this function with your own.

        :param obs: The input observations
        :return: The deterministic action of the actor
        """
        raise NotImplementedError

    def forward_critic1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Runs the first critic network.
        Override this function with your own.

        :param obs: The input observations
        :param action: The input actions
        :return: The output Q values of the critic network
        """
        raise NotImplementedError

    def forward_actor_target(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Runs the target actor network.
        Override this function with your own.

        :param obs: The input observations
        :return: The deterministic action of the actor
        """
        raise NotImplementedError

    def forward_critic_target1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Runs the first critic network.
        Override this function with your own.

        :param obs: The input observations
        :param action: The input actions
        :return: The output Q values of the critic network
        """
        raise NotImplementedError

    def update_targets(self) -> None:
        """
        Function to update the target networks periodically.
        Override this function with your own.
        """
        raise NotImplementedError

    def configure_optimizer(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Function to set up the optimizer.
        The first optimizer should be for the critics.
        The second should be the actor.
        Overide this function with your own.

        :return: The critic optimiser, followed by the actor optimiser
        """
        raise NotImplementedError
