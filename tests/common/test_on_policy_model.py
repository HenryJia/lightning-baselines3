import gym
import numpy as np
import pytest
from gym import spaces

from lightning_baselines3.common.on_policy_model import OnPolicyModel

from ..utils.env_checker import check_env

@pytest.mark.parametrize("env_id", ["CartPole-v0", "Pendulum-v0"])
def test_on_policy_model(env_id):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    model = OnPolicyModel(
        env_id,
        n_steps=100,
        batch_size=32,
        gamma=0.9,
        gae_lambda=0.95,
        ent_coef=0.1,
        vf_coef=1.0,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        buffer_grads=False,
        create_eval_env=False,
        monitor_wrapper=True,
        seed=None)
