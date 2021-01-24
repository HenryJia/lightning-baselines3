.. _dqn:

DQN
===

`Deep Q Network (DQN) <https://arxiv.org/abs/1312.5602>`_ builds on `Fitted Q-Iteration (FQI) <http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf>`_
and make use of different tricks to stabilize the learning with neural networks: it uses a replay buffer, a target network and gradient clipping.


Notes
-----

- Original paper: https://arxiv.org/abs/1312.5602
- Further reference: https://www.nature.com/articles/nature14236

.. note::
    This implementation provides only vanilla Deep Q-Learning and has no extensions such as Double-DQN, Dueling-DQN and Prioritized Experience Replay.
    It also does not include any speicifc policy so you can implement whichever one you like yourself. The example below shows epsilon greedy with a linearly decaying epsilon


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔      ✔
Box           ❌      ✔
MultiDiscrete ❌      ✔
MultiBinary   ❌      ✔
============= ====== ===========


Example
-------

.. literalinclude:: ../../../examples/dqn/minimal_cartpole.py

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

.. automodule:: lightning_baselines3.off_policy_models.dqn
.. autoclass:: DQN
  :members: