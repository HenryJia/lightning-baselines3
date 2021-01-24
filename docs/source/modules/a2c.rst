.. _a2c:

A2C
====

A synchronous, deterministic variant of `Asynchronous Advantage Actor Critic (A3C) <https://arxiv.org/abs/1602.01783>`_.
It uses multiple workers to avoid the use of a replay buffer.


.. warning::

  If you find training unstable or want to match performance of lightning-baselines A2C, consider using
  ``RMSpropTFLike`` optimizer from ``lightning_baselines3.common.sb2_compat.rmsprop_tf_like``.


Notes
-----

-  Original paper:  https://arxiv.org/abs/1602.01783
-  OpenAI blog post: https://openai.com/blog/baselines-acktr-a2c/


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

Train a A2C agent on ``CartPole-v1`` using 4 environments.

.. literalinclude:: ../../../examples/a2c/minimal_cartpole.py


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