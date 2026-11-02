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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()
    _, initial, reset, step = make_env(3, repeat=1)
    initial_batch = jax.tree.map(lambda x: jnp.repeat(x[None], args.envs, axis=0), initial)
    action = jnp.array([.6, .1])
    actions = jnp.tile(action, (args.envs, 1))
    scalar_step = jax.jit(lambda s: step(s, action)[0])
    batch_step = jax.jit(jax.vmap(lambda s: step(s, action)[0]))

    def serial():
        states = []
        for _ in range(args.envs):
            state = initial
            for _ in range(args.steps):
                state = scalar_step(state)
            states.append(state)
        return jax.tree.map(lambda *xs: jnp.stack(xs), *states)

    def vectorized():
        state = initial_batch
        for _ in range(args.steps):
            state = batch_step(state)
        return state

    @jax.jit
    def scanned(state):
        def body(state, _):
            return batch_step(state), None
        return jax.lax.scan(body, state, None, length=args.steps)[0]

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
            np.testing.assert_allclose(a, b, atol=2e-5)
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
              "methods": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
