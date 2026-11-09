"""Compare frozen checkpoints on the two matched variants of map 4."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from controller import demo_action
from env.robotaxi import RobotaxiEnv, swept_clearance
from evaluate_rl import evaluate
from ppo import load_checkpoint

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


ARTIFACTS = Path("artifacts")
VARIANTS = {"island": False, "cross_traffic": True}


def verify_layout(layouts):
    _, island = layouts["island"]
    p, traffic = layouts["cross_traffic"]
    for field in ("agent_pos", "goal_pos", "heading", "static_obstacles", "path_array"):
        np.testing.assert_array_equal(getattr(island, field), getattr(traffic, field))
    assert island.kinematic_obstacles.shape == (0, 2)
    assert traffic.kinematic_obstacles.shape == (1, 2)
    assert swept_clearance(island.agent_pos, island.goal_pos, island.static_obstacles,
                           island.static_obstacles, p.radius) <= 0
    np.testing.assert_array_equal(island.path_array[-1], island.goal_pos)


def drive_demo(p, initial, seed, episodes):
    """Keep the canonical start first, then use the same reset keys as PPO."""
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(seed, seed + episodes, dtype=jnp.int32))
    key = jax.random.PRNGKey(0)
    randomized = jax.vmap(lambda k: RobotaxiEnv.reset(k, p, initial, randomize=True)[1])(keys)
    states = jax.tree.map(lambda fixed, batch: jnp.concatenate((fixed[None], batch)),
                          initial, randomized)
    step = jax.vmap(lambda state, action: RobotaxiEnv.step(key, state, action, p))
    action = jax.vmap(lambda state: demo_action(state, p))

    @jax.jit
    def run(states):
        positions = jnp.zeros((p.max_steps_in_episode + 1, 2), dtype=states.agent_pos.dtype)
        positions = positions.at[0].set(states.agent_pos[1])
        batch_shape = states.time.shape

        def body(carry):
            tick, states, parked, crashed, gap, positions = carry
            _, states, _, _, info = step(states, action(states))
            return (tick + 1, states, parked | info["goal_reached"], crashed | info["collision"],
                    jnp.minimum(gap, info["clearance"]), positions.at[tick + 1].set(states.agent_pos[1]))

        return jax.lax.while_loop(
            lambda carry: ~jnp.all(carry[1].done), body,
            (0, states, jnp.zeros(batch_shape, bool), jnp.zeros(batch_shape, bool),
             jnp.full(batch_shape, jnp.inf, states.agent_pos.dtype), positions))

    _, states, parked, crashed, gap, positions = jax.tree.map(np.asarray, run(states))
    np.testing.assert_array_equal(parked.astype(int) + crashed + states.truncated, 1)
    distances = np.linalg.norm(states.agent_pos - states.goal_pos, axis=-1)
    assert (states.speed[parked] <= p.goal_speed).all()
    assert (distances[parked] <= p.goal_radius).all()
    rows = [
        {"seed": None if i == 0 else seed + i - 1,
         "outcome": "parked" if parked[i] else "collision" if crashed[i] else "timeout",
         "seconds": float(states.time[i] / p.fps), "clearance": float(gap[i]),
         "final_distance": float(distances[i]), "final_speed": float(states.speed[i])}
        for i in range(episodes + 1)
    ]
    return rows, positions[:states.time[1] + 1]


def plot_maps(output, layouts, demos, traces, seed):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.6), layout="constrained")
    fig.suptitle(f"Карта 4 · траектории на одном старте (seed {seed})", fontsize=16)
    for ax, (variant, (p, initial)) in zip(axes, layouts.items()):
        for pos in np.asarray(initial.static_obstacles):
            ax.add_patch(Circle(pos, p.radius, color="#627181", alpha=.8, lw=0))
        if VARIANTS[variant]:
            x, y = np.asarray(initial.kinematic_obstacles[0])
            ax.plot([x, x], [2, 43], color="#a43a65", lw=2, linestyle=":",
                    label="Линия движения трафика")
            ax.annotate("", xy=(x, y - 8), xytext=(x, y),
                        arrowprops={"arrowstyle": "-|>", "color": "#a43a65", "lw": 2})
        guide = np.asarray(initial.path_array)
        ax.plot(*guide.T, color="#8296aa", linestyle="--", label="Маршрут планировщика")
        ax.plot(*demos[variant].T, color="#087c68", lw=2, label="Демонстрационный водитель")
        path, outcome = traces[variant]
        ax.plot(*path.T, color="#bd4a18", lw=2, label="PPO карты 3")
        if outcome == "collision":
            ax.scatter(*path[-1], marker="x", s=75, color="#bd4a18", lw=2.5, zorder=6)
        ax.scatter(*initial.agent_pos, marker=">", color="#23344d", s=53, zorder=7)
        ax.scatter(*initial.goal_pos, marker="*", color="#087c68", s=130, zorder=7)
        for pos, label in [(initial.agent_pos, "Старт"), (initial.goal_pos, "Цель")]:
            ax.annotate(label, pos, xytext=(0, -13), textcoords="offset points", ha="center")
        ax.text(47.5, 11.7, "Островок", ha="center", color="#435363")
        title = "A · Объезд островка" if not VARIANTS[variant] else "B · Тот же объезд с поперечным трафиком"
        ax.set(title=title, aspect="equal", xlim=(-2, 97), ylim=(-2, 47), xticks=[], yticks=[])
        ax.legend(loc="lower left", ncols=2, fontsize=8, framealpha=.97)
    fig.savefig(output / "maps.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=110000)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--demo-episodes", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "map4_validation")
    args = parser.parse_args()
    if not 1 <= args.demo_episodes <= args.episodes:
        parser.error("Require 1 <= demo-episodes <= episodes")
    args.output.mkdir(parents=True, exist_ok=True)
    layouts = {
        name: RobotaxiEnv.init_params(jax.random.PRNGKey(0), 4, cross_traffic=traffic)
        for name, traffic in VARIANTS.items()
    }
    verify_layout(layouts)
    summary = {"seed_start": args.seed, "demo_episodes_per_variant": args.demo_episodes,
               "ppo_episodes_per_variant": args.episodes, "jax_version": jax.__version__,
               "physics_dtype": str(layouts["island"][1].agent_pos.dtype),
               "demo": {}, "ppo": {}}
    demos, traces = {}, {}
    for name, (p, initial) in layouts.items():
        rows, demos[name] = drive_demo(p, initial, args.seed, args.demo_episodes)
        summary["demo"][name] = rows
        print("Demo", name, sum(r["outcome"] == "parked" for r in rows), "/", len(rows), flush=True)

    checkpoints = json.loads((ARTIFACTS / "final_models.json").read_text())["checkpoints"]
    for source_map, name in checkpoints.items():
        path = ARTIFACTS / name
        params, config = load_checkpoint(path)
        results = {}
        targets = [(variant, 4, traffic) for variant, traffic in VARIANTS.items()]
        if source_map == "3":
            targets.insert(0, ("control_map3", 3, False))
        for variant, target_map, traffic in targets:
            result, trace = evaluate(params, config, target_map, episodes=args.episodes,
                                     seed=args.seed, cross_traffic=traffic)
            results[variant] = result
            counts = [result[key] for key in ("successes", "collisions", "timeouts")]
            print("PPO", source_map, variant, "parked/collision/timeout:", *counts, flush=True)
            if source_map == "3" and target_map == 4:
                first = result["per_episode"][0]
                steps = first["physics_steps"]
                repeat = result["control_repeat"]
                position = np.asarray(layouts[variant][1].agent_pos)
                route = np.vstack((position, trace["positions"][:(steps + repeat - 1) // repeat, 0]))
                outcome = "collision" if first["collision"] else "parked" if first["success"] else "timeout"
                traces[variant] = (route, outcome)
        summary["ppo"][source_map] = {"checkpoint": name,
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "results": results}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_maps(args.output, layouts, demos, traces, args.seed)


if __name__ == "__main__":
    main()
