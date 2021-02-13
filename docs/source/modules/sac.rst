.. _sac:

SAC
===

`Soft Actor Critic (SAC) <https://spinningup.openai.com/en/latest/algorithms/sac.html>`_ Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.

SAC is the successor of `Soft Q-Learning SQL <https://arxiv.org/abs/1702.08165>`_ and incorporates the double Q-learning trick from TD3.
A key feature of SAC, and a major difference with common RL algorithms, is that it is trained to maximize a trade-off between expected return and entropy, a measure of randomness in the policy.


Notes
-----

- Original paper: https://arxiv.org/abs/1801.01290
- OpenAI Spinning Guide for SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
- Original Implementation: https://github.com/haarnoja/sac
- Blog post on using SAC with real robots: https://bair.berkeley.edu/blog/2018/12/14/sac/

.. note::
    In our implementation, we use an entropy coefficient (as in OpenAI Spinning or Facebook Horizon),
    which is the equivalent to the inverse of reward scale in the original SAC paper.
    The main reason is that it avoids having too high errors when updating the Q functions.


.. note::

    The example model for SAC differ a bit from others: it uses ReLU instead of tanh activation,
    to match the original paper


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

.. literalinclude:: ../../../examples/sac/minimal_lunarlander.py

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

.. automodule:: lightning_baselines3.off_policy_models.sac
.. autoclass:: SAC
  :members: