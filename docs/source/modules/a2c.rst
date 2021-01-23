.. _a2c:

A2C
====

A synchronous, deterministic variant of `Asynchronous Advantage Actor Critic (A3C) <https://arxiv.org/abs/1602.01783>`_.
It uses multiple workers to avoid the use of a replay buffer.


.. warning::

  If you find training unstable or want to match performance of stable-baselines A2C, consider using
  ``RMSpropTFLike`` optimizer from ``stable_baselines3.common.sb2_compat.rmsprop_tf_like``.
  You can change optimizer with ``A2C(policy_kwargs=dict(optimizer_class=RMSpropTFLike, eps=1e-5))``.
  Read more `here <https://github.com/DLR-RM/stable-baselines3/pull/110#issuecomment-663255241>`_.


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

.. code-block:: python

    import gym

    import torch
    from  torch import distributions
    from torch import nn

    import pytorch_lightning as pl

    from lightning_baselines3.common.vec_env import make_vec_env, SubProcVecEnv
    from lightning_baselines3.on_policy_models import A2C


    class Model(A2C):
        def __init__(self, **kwargs):
            # **kwargs will pass our arguments on to PPO
            super(Model, self).__init__(**kwargs)

            self.actor = nn.Sequential(
                nn.Linear(self.observation_space.shape[0], 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.action_space.n),
                nn.Softmax(dim=1))

            self.critic = nn.Sequential(
                nn.Linear(self.observation_space.shape[0], 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1))

            self.save_hyperparameters()


        # This is for training the model, output the distribution and the corresponding value
        def forward(self, x):
            out = self.actor(x)
            dist = distributions.Categorical(probs=out)
            return dist, self.critic(x).flatten()


        # This is for inference and evaluation of our model
        def predict(self, x, deterministic=True):
            out = self.actor(x)
            if deterministic:
                out = torch.max(out, dim=1)[1]
            else:
                out = distributions.Categorical(probs=out).sample()
            return out.cpu().numpy()


        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
            return optimizer



    if __name__ == '__main__':
        env = make_vec_env('CartPole-v1', n_envs=4, vec_env_cls=SubProcVecEnv)
        eval_env = gym.make('CartPole-v1')
        model = Model(env=env, eval_env=eval_env)

        trainer = pl.Trainer(max_epochs=20, gradient_clip_val=0.5)
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

.. automodule:: lightning_baselines3.on_policy_models.a2c
.. autoclass:: A2C
  :members:
  :inherited-members: