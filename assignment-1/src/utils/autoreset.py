import jax
import chex
import equinox

from typing import Tuple

from env.base import BaseEnvParams, BaseEnvState, BaseEnv


class AutoResetWrapper(equinox.Module):
    env: BaseEnv = equinox.field(static=True)
    env_params: BaseEnvParams = equinox.field(static=True)
    init_state: BaseEnvState
    randomize: bool = equinox.field(static=True, default=False)

    @equinox.filter_jit
    def reset(
        self,
        key: chex.PRNGKey,
        *args
        # env_params: BaseEnvParams,
        # init_state: BaseEnvState | None,
    ) -> Tuple[chex.Array, BaseEnvState]:
        # Just call original reset
        if self.randomize:
            obs, state = self.env.reset(key, self.env_params, self.init_state, randomize=True)
        else:
            obs, state = self.env.reset(key, self.env_params, self.init_state)
        return obs, state

    @equinox.filter_jit
    def step(
        self,
        key: chex.PRNGKey,
        state: BaseEnvState,
        action: chex.Array,
        *args
        # env_params: BaseEnvParams,
    ) -> Tuple[chex.Array, BaseEnvState, chex.Scalar, BaseEnvState, dict]:
        """
        Returns: obs, state, reward, done, info
        But auto-resets if done=True
        """
        step_key, reset_key = jax.random.split(key)
        obs, new_state, reward, done, info = self.env.step(
            step_key, state, action, self.env_params
        )

        def reset_state(_):
            new_obs, new_st = self.reset(reset_key, self.env_params, self.init_state)
            # new_info = {"time": 0, "distance_to_path": new_obs.distance_to_path}
            # NOTE: we pass the reward from outside this context
            return new_obs, new_st, reward, info

        # If done, reset
        obs, new_state, reward, info = jax.lax.cond(
            done,
            reset_state,
            lambda _: (obs, new_state, reward, info),
            operand=None,
        )

        return obs, new_state, reward, done, info
