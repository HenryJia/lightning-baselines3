import os
import pytest

import gym
import numpy as np

from lightning_baselines3.common.vec_env import make_vec_env
from lightning_baselines3.common.monitor import Monitor
from lightning_baselines3.common.base_model import BaseModel

class DummyModel(BaseModel):
    def predict(self, x, deterministic=False):
        return np.array(self.action_space.sample()).reshape((1,))


def test_env_auto_monitor_wrap():
    env = gym.make("Pendulum-v0")
    model = BaseModel(env, env)
    assert not isinstance(model.env.envs[0], Monitor)

    env = Monitor(env)
    model = BaseModel(env, env)
    assert isinstance(model.env.envs[0], Monitor)

    model = BaseModel("Pendulum-v0", "Pendulum-v0")
    assert not isinstance(model.env.envs[0], Monitor)


def test_base_model_checks():
    """Test VecCheckNan Object"""

    env = make_vec_env("Pendulum-v0", n_envs=2)
    with pytest.raises(ValueError):
        model = BaseModel(env, "Pendulum-v0", support_multi_env=False)

    with pytest.raises(ValueError):
        model = BaseModel("CartPole-v1", "CartPole-v1", use_sde=True)


def test_evaluate_video():
    model = DummyModel("CartPole-v1", "CartPole-v1")
    model.evaluate(1, deterministic=True, render=False, record=True, record_fn='test_recording.mp4')

    assert os.path.isfile('test_recording.mp4')
    os.remove('test_recording.mp4')
