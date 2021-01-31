from typing import Optional, Tuple, Union

from gym import spaces

import torch
import torch.nn.functional as F

from lightning_baselines3.off_policy_models import OffPolicyModel
from lightning_baselines3.common.type_aliases import GymEnv


class TD3(OffPolicyModel):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

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
        super(TD3, self).__init__(
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
            squashed_actions=squashed_actions,
            use_sde=False,  # TD3 Does not support SDE since DQN only supports Discrete actions spaces
            use_sde_at_warmup=False,
            sde_sample_freq=-1)

        # We need manual optimization for this
        self.automatic_optimization = False

        assert isinstance(self.action_space, spaces.Box), "TD3 only supports environments with Box action spaces"

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.n_critics = 2  # Set this to 1 for DDPG

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

    def forward_critic2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Runs the second critic network.
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
        Runs the first target critic network.
        Override this function with your own.

        :param obs: The input observations
        :param action: The input actions
        :return: The output Q values of the critic network
        """
        raise NotImplementedError

    def forward_critic_target2(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Runs the second critic network.
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

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Specifies the update step for TD3. Override this if you wish to modify the TD3 algorithm
        """

        if self.num_timesteps < self.warmup_length:
            return

        opt_critic, opt_actor = self.optimizers(use_pl_optimizer=True)

        with torch.no_grad():
            noise = torch.randn_like(batch.actions) * self.target_policy_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = self.forward_actor_target(batch.next_observations) + noise

            # Compute the target Q value: min over all critics targets
            target1 = self.forward_critic_target1(batch.next_observations, next_actions)

            if self.n_critics == 2:
                target2 = self.forward_critic_target2(batch.next_observations, next_actions)
                target_q = torch.minimum(target1, target2)
            elif self.n_critics == 1:
                target_q = target1
            target_q = batch.rewards + (1 - batch.dones) * self.gamma * target_q

        # Get current Q estimates for each critic network
        current_q1 = self.forward_critic1(batch.observations, batch.actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q)
        self.log('critic_loss', critic_loss, on_step=True, prog_bar=True, logger=True)

        # Repeat for the other critic if we have 2
        if self.n_critics == 2:
            current_q2 = self.forward_critic2(batch.observations, batch.actions)
            critic_loss = critic_loss + F.mse_loss(current_q2, target_q)

        self.manual_backward(critic_loss, opt_critic)
        opt_critic.step()
        opt_critic.zero_grad()

        if batch_idx % self.policy_delay == 0:  # Optimize the actors
            # Compute actor loss
            actor_loss = -self.forward_critic1(batch.observations, self.forward_actor(batch.observations))
            actor_loss = actor_loss.mean()

            self.manual_backward(actor_loss, opt_actor)
            opt_actor.step()
            opt_actor.zero_grad()

            self.log('actor_loss', actor_loss, on_step=True, prog_bar=True, logger=True)

            self.update_targets()
