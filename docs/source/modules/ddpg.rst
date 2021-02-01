.. _ddpg:

DDPG
====

`Deep Deterministic Policy Gradient (DDPG) <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>`_ combines the
trick for DQN with the deterministic policy gradient, to obtain an algorithm for continuous actions.


.. note::

  As ``DDPG`` can be seen as a special case of its successor :ref:`TD3 <td3>`,
  they share the same policies and same implementation.


Notes
-----

- Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
- DDPG Paper: https://arxiv.org/abs/1509.02971
- OpenAI Spinning Guide for DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

.. note::

    DDPG by default uses a Tanh activated output like :ref:`TD3 <td3>`.
    DDPG by default expects actions bounded in [-1, 1], but this can be changed by setting :code:`squashed_action=False`


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Example
-------

.. literalinclude:: ../../../examples/ddpg/minimal_lunarlander.py

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

.. automodule:: lightning_baselines3.off_policy_models.ddpg
.. autoclass:: DDPG
  :members: