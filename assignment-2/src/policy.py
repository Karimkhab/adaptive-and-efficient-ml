"""Inference boundary: the learned actor accepts an observation vector only."""
import jax
import jax.numpy as jnp
from ppo import load_checkpoint, forward


class Policy:
    def __init__(self, path):
        self.params, self.config = load_checkpoint(path)
        self._predict = jax.jit(lambda obs: jnp.tanh(forward(self.params, obs)[0]))

    def __call__(self, observation):
        if observation.shape != (self.config["obs_dim"],):
            raise ValueError("Expected one observation v2 vector of length 70")
        return self._predict(jnp.asarray(observation, dtype=jnp.float32))
