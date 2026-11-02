"""Reproducible CPU PPO training; all updates and their real timings are logged."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from ppo import init_network, sample, forward, advantages, make_update, save_checkpoint
from rl_env import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    parser.add_argument("--map", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--reward-scale", type=float, default=.05)
    parser.add_argument("--heading-noise", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--entropy", type=float, default=.001)
    parser.add_argument("--route-reward", action="store_true")
    parser.add_argument("--parking-reward", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--cruise-speed", type=float, default=4.)
    parser.add_argument("--speed-cost", type=float, default=.1)
    parser.add_argument("--clearance-cost", type=float, default=0.)
    parser.add_argument("--clearance-margin", type=float, default=1.)
    parser.add_argument("--clearance-speed", action="store_true")
    args = parser.parse_args()
    if min(args.envs, args.rollout, args.minibatches, args.epochs, args.repeat, args.updates) < 1:
        parser.error("training dimensions must be positive")
    if args.learning_rate <= 0 or args.reward_scale <= 0 or args.heading_noise < 0 or args.entropy < 0:
        parser.error("learning rate/reward scale must be positive; noise/entropy must be nonnegative")
    if args.cruise_speed <= 0 or min(args.speed_cost,args.clearance_cost,args.clearance_margin) < 0:
        parser.error("cruise speed must be positive and safety costs/margin nonnegative")
    if args.envs * args.rollout % args.minibatches:
        parser.error("envs * rollout must be divisible by minibatches")
    if (args.output / "config.json").exists():
        parser.error("output already contains a run; choose a new --output to preserve its evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["init_checkpoint"] = str(args.init_checkpoint) if args.init_checkpoint else None
    config.update(gamma=.99, gae_lambda=.95, clip=.2, platform=platform.platform(),
                  python=platform.python_version(), jax=jax.__version__, devices=str(jax.devices()),
                  algorithm="PPO; tanh Gaussian; independent actor and critic",
                  observation_version=2, obs_dim=70, environment_commit="9f015e2cbadbfb24ea54a894d8db3e4b552c0554")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    p, initial, reset, step = make_env(args.map, args.repeat, args.max_steps, args.heading_noise,
        args.route_reward, args.randomize, args.parking_reward, args.cruise_speed,
        args.speed_cost, args.clearance_cost, args.clearance_margin, args.clearance_speed)
    reset_batch, step_batch = jax.vmap(reset), jax.vmap(step)
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    states, obs = jax.jit(reset_batch)(jax.random.split(reset_key, args.envs))
    params = init_network(args.seed, hidden=args.hidden)
    if args.init_checkpoint:
        from ppo import load_checkpoint
        params, initial_config = load_checkpoint(args.init_checkpoint)
        if initial_config["hidden"] != args.hidden:
            parser.error("--hidden must match the initialization checkpoint architecture")
    optimizer, update = make_update(args.learning_rate, args.epochs, args.minibatches, args.entropy)
    opt_state = optimizer.init(params)
    save_checkpoint(args.output / "untrained.npz", params, config)

    @jax.jit
    def collect(params, states, obs, episode_returns, key):
        def body(carry, _):
            states, obs, ep_returns, key = carry
            key, action_key, reset_key = jax.random.split(key, 3)
            action, latent, old_logp, value = sample(params, obs, action_key)
            final_states, final_obs, reward, done, info = step_batch(states, action)
            _, next_value = forward(params, final_obs)
            fresh_states, fresh_obs = reset_batch(jax.random.split(reset_key, args.envs))
            states = jax.tree.map(lambda end, fresh: jnp.where(
                done.reshape((args.envs,) + (1,) * (end.ndim - 1)), fresh, end), final_states, fresh_states)
            next_obs = jnp.where(done[:, None], fresh_obs, final_obs)
            ep_returns = ep_returns + reward
            records = (obs, latent, old_logp, value, reward * args.reward_scale,
                       next_value, info["success"] | info["collision"], done)
            stats = jnp.array([done.sum(), info["success"].sum(), info["collision"].sum(),
                               info["timeout"].sum(), jnp.where(done, ep_returns, 0).sum(),
                               info["physics_steps"].sum(), reward.sum()])
            return (states, next_obs, jnp.where(done, 0., ep_returns), key), (records, stats)

        carry, (records, stats) = jax.lax.scan(body, (states, obs, episode_returns, key), None, length=args.rollout)
        ob, latent, lp, val, rew, nv, terminal, done = records
        adv, target = advantages(rew, val, nv, terminal, done)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = tuple(x.reshape((-1,) + x.shape[2:]) for x in (ob, latent, lp, val, adv, target))
        return carry, batch, stats.sum(axis=0)

    fields = ["update", "decisions", "physics_steps", "episodes", "successes", "collisions", "timeouts",
              "episode_return_mean", "reward_sum", "loss", "actor_loss", "critic_loss", "latent_entropy",
              "approx_kl", "clip_fraction", "collect_seconds", "update_seconds", "elapsed_seconds"]
    episode_returns = jnp.zeros(args.envs)
    physics_steps = 0
    history = []
    start = time.perf_counter()
    with (args.output / "training.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for iteration in range(1, args.updates + 1):
            t = time.perf_counter()
            (states, obs, episode_returns, key), batch, stats = collect(params, states, obs, episode_returns, key)
            jax.block_until_ready(stats)
            collect_time = time.perf_counter() - t
            key, update_key = jax.random.split(key)
            t = time.perf_counter()
            params, opt_state, losses = update(params, opt_state, batch, update_key)
            jax.block_until_ready(losses)
            update_time = time.perf_counter() - t
            stats, losses = np.asarray(stats), np.asarray(losses)
            if not np.isfinite(losses).all():
                raise RuntimeError(f"Nonfinite loss at update {iteration}")
            physics_steps += int(stats[5])
            row = dict(zip(fields, [iteration, iteration * args.envs * args.rollout, physics_steps,
                *map(int, stats[:4]), float(stats[4] / max(stats[0], 1)), float(stats[6]),
                *map(float, losses), collect_time, update_time, time.perf_counter() - start]))
            writer.writerow(row)
            file.flush()
            history.append(row)
            if iteration == 1 or iteration % 10 == 0:
                recent = history[-10:]
                episodes = sum(r["episodes"] for r in recent)
                successes = sum(r["successes"] for r in recent)
                print(json.dumps({"update": iteration, "decisions": row["decisions"],
                    "episodes_last10": episodes, "success_rate_last10": successes / max(episodes, 1),
                    "collect_s": round(collect_time, 3), "elapsed_s": round(row["elapsed_seconds"], 1)}), flush=True)
            if iteration % 50 == 0 or iteration == args.updates:
                metadata = dict(config, completed_updates=iteration, physics_steps=physics_steps,
                                decisions=row["decisions"], elapsed_seconds=row["elapsed_seconds"])
                save_checkpoint(args.output / "checkpoint.npz", params, metadata)
                if iteration % 50 == 0:
                    save_checkpoint(args.output / f"checkpoint_{iteration:04}.npz", params, metadata)


if __name__ == "__main__":
    main()
