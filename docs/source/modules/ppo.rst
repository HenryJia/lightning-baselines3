.. _ppo2:

PPO
===

The `Proximal Policy Optimization <https://arxiv.org/abs/1707.06347>`_ algorithm combines ideas from A2C (having multiple workers)
and TRPO (it uses a trust region to improve the actor).

The main idea is that after an update, the new policy should be not too far form the old policy.
For that, ppo uses clipping to avoid too large update.


.. note::

  PPO contains several modifications from the original algorithm not documented
  by OpenAI: advantages are normalized and value function can be also clipped .


Notes
-----

- Original paper: https://arxiv.org/abs/1707.06347
- Clear explanation of PPO on Arxiv Insights channel: https://www.youtube.com/watch?v=5P7I-xPq8u8
- OpenAI blog post: https://blog.openai.com/openai-baselines-ppo/
- Spinning Up guide: https://spinningup.openai.com/en/latest/algorithms/ppo.html


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ✔️      ✔️
MultiDiscrete ✔️      ✔️
MultiBinary   ✔️      ✔️
============= ====== ===========


Example
-------

Train a PPO agent on ``CartPole-v1`` using 4 environments.

.. literalinclude:: ../../../examples/ppo/minimal_cartpole.py

Results
-------

Atari Games
^^^^^^^^^^^

Coming soon


How to replicate the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon

Parameters
----------

.. automodule:: lightning_baselines3.on_policy_models.a2c
.. autoclass:: A2C
  :members: