import gym
import numpy as np
import pytest
from gym import spaces

from ..utils.env_checker import check_env

@pytest.mark.parametrize("env_id", ["CartPole-v0", "Pendulum-v0"])
def test_on_policy_model(env_id):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    env = gym.make(env_id)
    with pytest.warns(None) as record:
        check_env(env)

    # Pendulum-v0 will produce a warning because the action space is
    # in [-2, 2] and not [-1, 1]
    if env_id == "Pendulum-v0":
        assert len(record) == 1
    else:
        # The other environments must pass without warning
        assert len(record) == 0
