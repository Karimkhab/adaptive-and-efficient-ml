"""Equal-work CPU benchmark; separate compile time and synchronized execution."""
import argparse
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

from rl_env import make_env
from env.robotaxi import RobotaxiEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()
    if min(args.envs, args.steps, args.repeats) < 1:
        parser.error("envs, steps and repeats must be positive")
    params, initial, _, step = make_env(3, repeat=1)
    action = jnp.array([.6, .1])
    initial_carry = (initial, jnp.zeros(RobotaxiEnv.observation_size(params), dtype=jnp.float32),
                     jnp.zeros_like(initial.speed), jnp.zeros_like(initial.time))
    initial_batch = jax.tree.map(lambda x: jnp.repeat(x[None], args.envs, axis=0), initial_carry)

    def transition(carry):
        state, observations, rewards, endings = carry
        state, obs, reward, done, _ = step(state, action)
        # Consume every step's outputs so the compiler must compute the
        # observation and reward even when only final statistics are returned.
        return state, observations + obs, rewards + reward, endings + done

    scalar_step = jax.jit(transition)
    batch_step = jax.jit(jax.vmap(transition))

    def serial():
        results = []
        for _ in range(args.envs):
            carry = initial_carry
            for _ in range(args.steps):
                carry = scalar_step(carry)
            results.append(carry)
        return jax.tree.map(lambda *xs: jnp.stack(xs), *results)

    def vectorized():
        carry = initial_batch
        for _ in range(args.steps):
            carry = batch_step(carry)
        return carry

    @jax.jit
    def scanned(carry):
        def body(carry, _):
            return batch_step(carry), None
        return jax.lax.scan(body, carry, None, length=args.steps)[0]

    functions = {"python_serial_jit_step": serial, "python_loop_vmap": vectorized,
                 "jit_scan_vmap": lambda: scanned(initial_batch)}
    rows = {}
    reference = None
    for name, fn in functions.items():
        start = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        cold = time.perf_counter() - start
        if reference is None:
            reference = result
        for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(reference)):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=2e-5)
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = fn()
            jax.block_until_ready(result)
            timings.append(time.perf_counter() - start)
        median = float(np.median(timings))
        rows[name] = {"cold_seconds_including_compile": cold, "warm_seconds": timings,
                      "warm_median_seconds": median,
                      "physics_transitions_per_second": args.envs * args.steps / median}
        print(name, rows[name], flush=True)
    output = {"platform": platform.platform(), "jax": jax.__version__, "devices": str(jax.devices()),
              "map": 3, "envs": args.envs, "steps": args.steps, "action_repeat": 1,
              "transitions_per_trial": args.envs * args.steps, "equal_final_states": True,
              "equal_observation_reward_and_done_totals": True,
              "work": "environment state, observation, reward and done; fixed actions, no PPO",
              "methods": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
