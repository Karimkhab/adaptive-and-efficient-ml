"""Reproducible, deterministic controller rollouts. No model is trained here."""
import argparse
import json
import platform
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from controller import demo_action
from env.robotaxi import RobotaxiEnv


def rollout(map_id, seed=0, max_steps=3000, on_frame=None, params=None, initial_state=None):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    key = jax.random.PRNGKey(seed)
    if params is None:
        p, initial = RobotaxiEnv.init_params(key, map_id=map_id, max_steps=max_steps)
    else:
        p, initial = params, initial_state
    obs, state = RobotaxiEnv.reset(key, p, initial)
    # Separate first compilation from execution timing.
    demo_action(state, p).block_until_ready()
    jax.block_until_ready(RobotaxiEnv.step(key, state, jnp.zeros(2), p))
    rows = []
    episode_return = 0.
    start = time.perf_counter()
    for step in range(max_steps):
        action = demo_action(state, p)
        obs, state, reward, done, info = RobotaxiEnv.step(key, state, action, p)
        if not bool(jnp.all(jnp.isfinite(obs.as_vector()))):
            raise RuntimeError("Non-finite observation")
        episode_return += float(reward)
        rows.append([int(state.time), *np.asarray(state.agent_pos).tolist(),
                     float(state.heading), float(state.speed),
                     *np.asarray(action).tolist(), float(reward), float(info["clearance"])])
        if on_frame is not None:
            on_frame(state, obs, dict(info, action=action), p, map_id)
        if bool(done):
            break
    elapsed = time.perf_counter() - start
    result = {
        "map_id": map_id, "seed": seed, "steps": len(rows),
        "simulation_seconds": len(rows) * p.step_size,
        "goal_reached": bool(info["goal_reached"]), "collision": bool(info["collision"]),
        "truncated": bool(info["truncated"]), "return": episode_return,
        "final_distance": float(info["distance_to_goal"]),
        "final_speed": float(state.speed),
        "goal_speed_limit": p.goal_speed,
        "min_clearance": min(r[-1] for r in rows),
        "max_speed": max(r[4] for r in rows), "execution_seconds": elapsed,
    }
    return result, np.asarray(rows), p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for map_id in args.maps:
        result, trajectory, _ = rollout(map_id, args.seed, args.max_steps)
        results.append(result)
        np.savetxt(args.output / f"trajectory_map{map_id}.csv", trajectory, delimiter=",",
                   header="step,x,y,heading,speed,acceleration_action,steering_action,reward,clearance", comments="")
        print(json.dumps(result), flush=True)
    metadata = {
        "controller": "deterministic heuristic, privileged state; no RL training",
        "python": platform.python_version(), "platform": platform.platform(),
        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "observation_version": RobotaxiEnv.OBSERVATION_VERSION,
        "environment_revision": RobotaxiEnv.ENVIRONMENT_REVISION,
        "observation_size": 70,
        "results": results,
    }
    (args.output / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
