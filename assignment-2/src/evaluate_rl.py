"""Evaluate frozen policies without autoreset or training-time rewards."""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

from ppo import load_checkpoint, init_network, forward, sample, control_repeat
from rl_env import make_env
from env.robotaxi import DEFAULT_HEADING_NOISE, RobotaxiEnv


def evaluate(params, config, map_id, episodes=32, seed=2000, heading_noise=None,
             policy="trained", stochastic=False, randomize=True, variants=False):
    if heading_noise is None:
        heading_noise = DEFAULT_HEADING_NOISE if randomize else 0.
    if not math.isfinite(heading_noise) or heading_noise < 0:
        raise ValueError("heading noise must be finite and nonnegative")
    repeat=control_repeat(config)
    p, initial, reset, step = make_env(map_id, repeat, config["max_steps"], heading_noise,
                                     randomize=randomize)
    reset_batch, step_batch = jax.vmap(reset), jax.vmap(step)
    if variants:
        from scenarios import VARIANTS, vary_initial_state
        episodes = len(VARIANTS)
        cases = [vary_initial_state(p, initial, **settings) for name, settings in VARIANTS]
        fixed_states = jax.tree.map(lambda *xs: jnp.stack(xs), *cases)

    @jax.jit
    def run(params, key):
        keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(seed, seed + episodes, dtype=jnp.int32))
        if variants:
            states = fixed_states
            obs = jax.vmap(lambda s: RobotaxiEnv.get_observation(s, p).as_vector())(states)
        else:
            states, obs = reset_batch(keys)

        def body(carry, _):
            states, obs, key = carry
            key, action_key = jax.random.split(key)
            if policy == "random":
                action = jax.random.uniform(
                    action_key, (episodes, 2), dtype=obs.dtype, minval=-1., maxval=1.)
            elif stochastic:
                action, _, _, _ = sample(params, obs, action_key)
            else:
                mean, _ = forward(params, obs)
                action = jnp.tanh(mean)
            states, obs, reward, done, info = step_batch(states, action)
            return (states, obs, key), (reward, info, states.agent_pos, states.heading)

        (states, _, _), (reward, info, positions, headings) = jax.lax.scan(
            body, (states, obs, key), None, length=(config["max_steps"] + repeat - 1) // repeat)
        return states, reward, info, positions, headings

    start = time.perf_counter()
    states, rewards, info, positions, headings = run(params, jax.random.PRNGKey(seed + 99999))
    jax.block_until_ready(rewards)
    elapsed = time.perf_counter() - start
    rewards, info = np.asarray(rewards), jax.tree.map(np.asarray, info)
    success = info["success"].any(axis=0)
    collision = info["collision"].any(axis=0)
    timeout = info["timeout"].any(axis=0)
    returns = rewards.sum(axis=0)
    np.testing.assert_array_equal(success.astype(int) + collision + timeout, np.ones(episodes))
    assert np.isfinite(positions).all() and np.isfinite(returns).all()
    assert np.all(np.asarray(states.speed)[success] <= p.goal_speed)
    assert np.all(info["distance"][-1, success] <= p.goal_radius)
    initial_distance = float(jnp.linalg.norm(initial.agent_pos - initial.goal_pos))
    reconstructed = (initial_distance - info["distance"][-1] - p.time_cost * np.asarray(states.time) / p.fps
                     + p.goal_bonus * success - p.collision_penalty * collision)
    # Near-stationary norm subtraction can accumulate a few ULPs per
    # physical step; base the bound on the simulator precision.
    # This changes only the independent consistency assertion, never rewards
    # or the exact parking/collision/timeout decisions.
    distance_scale = p.cell_size * math.hypot(*p.map_height_width)
    distance_ulp = float(np.spacing(rewards.dtype.type(distance_scale)))
    return_tolerance=np.maximum(.002,4*distance_ulp*np.asarray(states.time))
    assert np.all(np.abs(returns-reconstructed)<=return_tolerance), (returns,reconstructed,return_tolerance)
    summary = {"map": map_id, "policy": policy, "stochastic": stochastic,
               "episodes": episodes, "seed_start": seed, "physics_dtype": str(initial.agent_pos.dtype),
               "reset_distribution": ({"fixed_variants": [{"name": name, "settings": settings} for name, settings in VARIANTS]} if variants else ({"heading_radians": [-heading_noise, heading_noise],
                   "speed": [0., .5], "traffic_speed_multiplier": [.8, 1.2], "traffic_phase_steps": [0, p.fps]}
                   if randomize else {"heading_radians": [-heading_noise, heading_noise], "other_state": "canonical"})),
               "randomize_all": randomize and not variants, "observation_version": RobotaxiEnv.OBSERVATION_VERSION,
               "environment_revision": RobotaxiEnv.ENVIRONMENT_REVISION,
               "variant_suite": variants,
               "training_repeat":config['repeat'],"control_repeat":repeat,
               "control_frequency_hz":p.fps/repeat,
               "successes": int(success.sum()), "collisions": int(collision.sum()),
               "timeouts": int(timeout.sum()), "success_rate": float(success.mean()),
               "return_mean": float(returns.mean()), "return_std": float(returns.std()),
               "distance_mean": float(info["distance"][-1].mean()),
               "episode_sim_seconds_mean": float(np.asarray(states.time).mean() / p.fps),
               "min_clearance": float(info["clearance"].min()),
               "final_speed_max": float(np.asarray(states.speed).max()),
               "raw_return_reconstruction_max_error": float(np.max(np.abs(returns - reconstructed))),
               "raw_return_reconstruction_max_tolerance":float(return_tolerance.max()),
               "evaluation_seconds_including_compile": elapsed,
               "per_episode": [{"index": i, "reset_seed": seed + i if not variants else None,
                   "variant": VARIANTS[i][0] if variants else None, "success": bool(success[i]),
                   "collision": bool(collision[i]), "timeout": bool(timeout[i]),
                   "return": float(returns[i]), "final_distance": float(info["distance"][-1, i]),
                   "final_speed": float(states.speed[i]),
                   "physics_steps": int(states.time[i])} for i in range(episodes)]}
    return summary, {"positions": np.asarray(positions), "headings": np.asarray(headings),
                     "returns": returns, "success": success}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--maps", type=int, nargs="+", default=[3])
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2000)
    reset_mode = parser.add_mutually_exclusive_group()
    reset_mode.add_argument("--heading-noise", type=float,
                            help="Maximum heading offset in radians (default: pi/18)")
    parser.add_argument("--policy", choices=["trained", "untrained", "random"], default="trained")
    parser.add_argument("--stochastic", action="store_true")
    reset_mode.add_argument("--canonical", action="store_true")
    reset_mode.add_argument("--variants", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    if args.heading_noise is not None and (not math.isfinite(args.heading_noise) or args.heading_noise < 0):
        parser.error("heading noise must be finite and nonnegative")
    params, config = load_checkpoint(args.checkpoint)
    if args.policy == "untrained":
        params = init_network(config["seed"], hidden=config["hidden"])
    summaries = []
    for map_id in args.maps:
        result, traces = evaluate(params, config, map_id, args.episodes, args.seed,
                                  0. if args.canonical else args.heading_noise,
                                  args.policy, args.stochastic, randomize=not args.canonical,
                                  variants=args.variants)
        summaries.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "per_episode"}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_name(args.output.stem + f"_map{map_id}.npz"), **traces)
    args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training_config": config, "results": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
