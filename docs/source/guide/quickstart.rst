.. _quickstart:

===============================
Lightning Baselines3 in 3 Steps
===============================
***************************
Step 1: Choose an algorithm
***************************

We will use A2C in this example.

.. code-block:: python

    # Minimal Example for the CartPole-v1 environment with PPO
    import gym

    import torch
    from  torch import distributions
    from torch import nn

    import pytorch_lightning as pl

    from lightning_baselines3.on_policy_models import A2C

*************************
Step 2: Define Your Model
*************************

.. code-block:: python

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


**********************************
Step 2: Fit with Lightning Trainer
**********************************

.. code-block:: python

    if __name__ == '__main__':
        env = gym.make('CartPole-v1') # Make the environment
        model = Model(env=env, eval_env=env) # Use that environment for training and evaluation

        # Add some gradient clipping for good measure
        trainer = pl.Trainer(max_epochs=20, gradient_clip_val=0.5)
        trainer.fit(model)

        # Evaluate
        model.evaluate(num_eval_episodes=10, render=True)