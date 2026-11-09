"""Build figures from saved training logs and evaluation outcomes."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import jax
from env.robotaxi import RobotaxiEnv


def training(folder, limit=None):
    with (Path("artifacts") / folder / "training.csv").open() as file:
        rows = list(csv.DictReader(file))[:limit]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def curve(ax, data, label, color):
    n = 10
    total = np.convolve(data["episodes"], np.ones(n), mode="valid")
    successes = np.convolve(data["successes"], np.ones(n), mode="valid")
    rate = np.divide(successes, total, out=np.zeros_like(total), where=total > 0)
    ax.plot(data["decisions"][n-1:] / 1e6, rate, label=label, color=color)


def recheck_figures(out):
    current = json.loads(Path("artifacts/recheck/summary.json").read_text())
    holdout = current["holdout"]
    totals = []
    for map_id in range(4):
        rows = [r for r in holdout if r["map"] == map_id]
        totals.append([sum(r[key] for r in rows) for key in ("successes", "episodes")])

    teal, orange, slate = "#15806c", "#b44c2b", "#668296"
    fig, (parking, collisions) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    parking.barh(range(4), [100 * k / n for k, n in totals], color=teal, height=.55)
    for y, (k, n) in enumerate(totals):
        parking.text(103, y, f"{k} / {n}", va="center")
    parking.set(yticks=range(4), yticklabels=[f"Карта {m}" for m in range(4)],
                xlabel="Успешные парковки, %", title="Парковки на новых стартах",
                xlim=(0, 124), xticks=[0, 25, 50, 75, 100])
    parking.invert_yaxis()
    parking.grid(axis="x", alpha=.2)

    pairs = current["map2_matched_comparison"]
    x = np.arange(len(pairs))
    for key, label, offset, color in [
        ("published", "Версия от 2 ноября", -.18, slate),
        ("revised", "Текущая версия", .18, orange),
    ]:
        bars = collisions.bar(x + offset, [r[key]["collisions"] for r in pairs], .34,
                              label=label, color=color)
        collisions.bar_label(bars, padding=3)
    collisions.set(title="Карта 2 на одинаковых стартах",
                   ylabel="Столкновения из 2048 поездок",
                   xticks=x, xticklabels=[f'{a}–{b}' for a, b in (r["seed_range"] for r in pairs)],
                   xlabel="Диапазон seed", ylim=(0, 10))
    collisions.grid(axis="y", alpha=.2)
    collisions.legend(frameon=False, loc="upper left")
    fig.savefig(out / "recheck.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.6), constrained_layout=True)
    models = json.loads(Path("artifacts/final_models.json").read_text())["checkpoints"]
    for map_id, color in [(1, "#2467a6"), (2, teal)]:
        checkpoint = Path(models[str(map_id)])
        config = json.loads((Path("artifacts") / checkpoint.with_suffix(".json")).read_text())
        data = training(checkpoint.parent)
        curve(ax, data, f"Карта {map_id}", color)
        selected = data["decisions"][config["completed_updates"] - 1] / 1e6
        ax.axvline(selected, linestyle="--", color=color, alpha=.65)
    ax.set(title="Дообучение карт 1 и 2 в исправленной среде",
           xlabel="Решения с начала дообучения, млн",
           ylabel="Доля парковок за 10 обновлений", ylim=(-.03, 1.03))
    ax.grid(alpha=.2)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "finetuning.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recheck-only", action="store_true", help="Generate only the current figures")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    out = Path("artifacts/figures")
    out.mkdir(parents=True, exist_ok=True)
    recheck_figures(out)
    if args.recheck_only:
        return
    colors = ["#2467a6", "#168765", "#bd8619", "#b94747"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed in range(3):
        curve(axes[0], training(f"ppo_v2_map3_seed{seed}", 500), f"shaped / seed {seed}", colors[seed])
    curve(axes[0], training("ppo_v2_base_map3_seed0"), "base / seed 0", colors[3])
    for folder, label, color in [("ppo_v2_map0_seed0", "route + parking / seed 0", colors[0]),
                                  ("ppo_v2_base_map0_seed0", "base / seed 0", colors[3])]:
        curve(axes[1], training(folder), label, color)
    for ax, title in zip(axes, ["Map 3 / three initializations + baseline", "Map 0 / reward ablation"]):
        ax.set(title=title, xlabel="Policy decisions (millions)", ylabel="Parked / completed training episodes", ylim=(-.03, 1.03))
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(out / "learning.png")
    fig.savefig(out / "learning.svg")
    plt.close(fig)
    rows = json.loads(Path("artifacts/evaluation/summary.json").read_text())["results"]
    lookup = {r["name"]: r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for map_id, color in enumerate(colors):
        r = lookup[f"map{map_id}_final"]
        axes[0].bar(map_id, r["success_rate"], color=color)
        axes[0].text(map_id, r["success_rate"] + .02, f'{r["successes"]}/{r["episodes"]}', ha="center")
    axes[0].set(title="Final deterministic policies / held-out resets", xticks=range(4),
                xticklabels=["map 0", "map 1*", "map 2", "map 3"], ylim=(0, 1.15), ylabel="Parking success")
    names = ["map0_base", "map0_shaped500", "map3_base", "map3_shaped500"]
    for x, name in enumerate(names):
        r = lookup[name]
        axes[1].bar(x, r["success_rate"], color=colors[3] if "base" in name else colors[0])
        axes[1].text(x, r["success_rate"] + .02, f'{r["successes"]}/{r["episodes"]}', ha="center")
    axes[1].set(title="Equal budget / 500 PPO updates / seed 0", xticks=range(4),
                xticklabels=["map0\nbase", "map0\nshaped", "map3\nbase", "map3\nshaped"], ylim=(0, 1.15))
    fig.suptitle("Observation v2 / parked: distance <=1, speed <=0.001 / * map1 uses PPO transfer")
    fig.savefig(out / "evaluation.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for map_id, ax in enumerate(axes.ravel()):
        p, state = RobotaxiEnv.init_params(jax.random.PRNGKey(0), map_id=map_id)
        for pos in np.asarray(state.static_obstacles):
            ax.add_patch(Circle(pos, p.radius, color="#657280"))
        for pos in np.asarray(state.kinematic_obstacles):
            ax.add_patch(Circle(pos, p.radius, color="#d67c4a", alpha=.65))
        with np.load(f"artifacts/evaluation/map{map_id}_final_trajectories.npz") as data:
            positions = data["positions"]
        for i in range(positions.shape[1]):
            ax.plot(positions[:, i, 0], positions[:, i, 1], color=colors[map_id], alpha=.12, lw=.7)
        ax.scatter(*state.agent_pos, marker="o", color=colors[map_id], s=35)
        ax.scatter(*state.goal_pos, marker="*", color="#12805e", s=100)
        r = lookup[f"map{map_id}_final"]
        ax.set(title=f'Map {map_id}: {r["successes"]}/{r["episodes"]} parked; all trajectories',
               xlim=(-2, 97), ylim=(-2, 47), aspect="equal", xlabel="x / world units", ylabel="y / world units")
    fig.suptitle("Moving obstacles shown at canonical t=0; trajectories use randomized traffic")
    fig.savefig(out / "trajectories.png")
    fig.savefig(out / "trajectories.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
