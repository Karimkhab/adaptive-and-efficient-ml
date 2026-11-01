"""Tiny observation-only rollout. This checks the RL interface, not learning."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import jax
import jax.numpy as jnp
from env.robotaxi import RobotaxiEnv
from utils.autoreset import AutoResetWrapper


@jax.jit
def policy(key, observation):
    # A replaceable example policy receives only the observation vector.
    steer = jnp.clip(jnp.arctan2(observation[2], observation[1]), -1., 1.)
    gas = jnp.where(observation[0] < .3, .4, -.3)
    return jnp.array([gas, steer]) + jax.random.uniform(key, (2,), minval=-.02, maxval=.02)


def main():
    key = jax.random.PRNGKey(123)
    params, initial = RobotaxiEnv.init_params(key, map_id=0)
    env = AutoResetWrapper(RobotaxiEnv, params, initial, randomize=True)
    obs, state = env.reset(key)
    episodes = 0
    for _ in range(128):
        key, policy_key, env_key = jax.random.split(key, 3)
        action = jnp.clip(policy(policy_key, obs.as_vector()), -1., 1.)
        obs, state, reward, done, info = env.step(env_key, state, action)
        successor = jnp.where(done, info['final_observation'], obs.as_vector())
        # Illustrative one-step value target; only true termination masks bootstrap.
        target = reward + .99 * (1 - info['terminated']) * jnp.sum(successor)
        assert obs.as_vector().shape == (70,) and bool(jnp.isfinite(target))
        assert bool(jnp.all(jnp.isfinite(obs.as_vector())))
        episodes += int(done)
    print(f'Observation v2: 128 finite transitions, {episodes} ended episodes; no training performed')


if __name__ == '__main__':
    main()
