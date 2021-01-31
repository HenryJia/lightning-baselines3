.. _td3:

TD3
===

`Twin Delayed DDPG (TD3) <https://spinningup.openai.com/en/latest/algorithms/td3.html>`_ Addressing Function Approximation Error in Actor-Critic Methods.

TD3 is a direct successor of :ref:`DDPG <ddpg>` and improves it using three major tricks: clipped double Q-Learning, delayed policy update and target policy smoothing.
We recommend reading `OpenAI Spinning guide on TD3 <https://spinningup.openai.com/en/latest/algorithms/td3.html>`_ to learn more about those.


Notes
-----

- Original paper: https://arxiv.org/pdf/1802.09477.pdf
- OpenAI Spinning Guide for TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
- Original Implementation: https://github.com/sfujim/TD3

.. note::

    The original TD3 paper uses a Tanh activated output. This example does the same.
    TD3 by default expects actions bounded in [-1, 1], but this can be changed by setting :code:`squashed_action=False`


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

.. literalinclude:: ../../../examples/td3/minimal_lunarlander.py

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

.. automodule:: lightning_baselines3.off_policy_models.td3
.. autoclass:: TD3
  :members: