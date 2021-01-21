import os
import shutil

import gym
import numpy as np

from lightning_baselines3.common.monitor import Monitor
from lightning_baselines3.common.base_model import BaseModel


def test_env_auto_monitor_wrap():
    env = gym.make("Pendulum-v0")
    model = BaseModel(env, env)
    assert not isinstance(model.env.envs[0], Monitor)

    env = Monitor(env)
    model = BaseModel(env, env)
    assert isinstance(model.env.envs[0], Monitor)

    model = BaseModel("Pendulum-v0", "Pendulum-v0")
    assert not isinstance(model.env.envs[0], Monitor)
