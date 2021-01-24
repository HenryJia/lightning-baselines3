.. _dqn:

DQN
===

`Deep Q Network (DQN) <https://arxiv.org/abs/1312.5602>`_ builds on `Fitted Q-Iteration (FQI) <http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf>`_
and make use of different tricks to stabilize the learning with neural networks: it uses a replay buffer, a target network and gradient clipping.


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    CnnPolicy


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

.. code-block:: python

    import copy
    import gym

    import torch
    from torch import nn

    import pytorch_lightning as pl

    from lightning_baselines3.off_policy_models import DQN


    class Model(DQN):
        def __init__(self, **kwargs):
            # **kwargs will pass our arguments on to DQN
            super(Model, self).__init__(**kwargs)

            self.qnet = nn.Sequential(
                nn.Linear(self.observation_space.shape[0], 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.action_space.n))

            self.qnet_target = copy.deepcopy(self.qnet)

            self.eps = 1.0
            self.eps_init = 1.0
            self.eps_decay = 5000
            self.eps_final = 0.05

            self.qnet_target = copy.deepcopy(self.qnet)

            self.save_hyperparameters()


        # This is for running the model, returns the Q values given our observation
        def forward(self, x):
            return self.qnet(x)


        # This is for running the target Q network
        def forward_target(self, x):
            return self.qnet_target(x)


        # This is for updating the target Q network
        def update_target(self):
            self.qnet_target.load_state_dict(self.qnet.state_dict())


        # Use the environment step callback to linearly decay our
        # epsilon per envrionment step for epsilon greedy
        def on_step(self):
            k = max(self.eps_decay - self.num_timesteps, 0) / self.eps_decay
            self.eps = self.eps_final + k * (self.eps_init - self.eps_final)


        # This is for inference and evaluation of our model, returns the action
        def predict(self, x, deterministic=True):
            out = self.qnet(x)
            if deterministic:
                out = torch.max(out, dim=1)[1]
            else:
                eps = torch.rand_like(out[:, 0])
                eps = (eps < self.eps).float()
                out = eps*torch.rand_like(out).max(dim=1)[1] + (1-eps)*out.max(dim=1)[1]
            return out.long().cpu().numpy()


        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
            return optimizer


    if __name__ == '__main__':
        model = Model(env='CartPole-v1', eval_env='CartPole-v1')

        trainer = pl.Trainer(max_epochs=10, gradient_clip_val=0.5)
        trainer.fit(model)

        model.evaluate(num_eval_episodes=10, render=True)

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