"""Frozen evaluation: held-out resets, endpoint scenarios, baselines and transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import numpy as np

from evaluate_rl import evaluate
from ppo import load_checkpoint, init_network
from make_rl_video import FINAL_UPDATES, checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--map2-episodes", type=int, default=512)
    parser.add_argument("--map2-seed", type=int, default=50000)
    args = parser.parse_args()
    out = args.root / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def run(name, path, map_id, policy="trained", stochastic=False, canonical=False, variants=False,
            episodes=None,seed=None):
        params, config = load_checkpoint(path)
        if policy == "untrained":
            params = init_network(config["seed"], hidden=config["hidden"])
        n=args.episodes if episodes is None else episodes
        seed=args.seed if seed is None else seed
        result, traces = evaluate(params, config, map_id, 1 if canonical else n,
            seed, 0., policy, stochastic, randomize=not canonical, variants=variants)
        result.update(name=name, checkpoint=str(path), train_seed=config["seed"],
                      completed_updates=config["completed_updates"],
                      checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(out / f"{name}_trajectories.npz", **traces)
        rows.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "per_episode"}), flush=True)
        jax.clear_caches()

    for map_id in FINAL_UPDATES:
        # map1 transfer is an explicit extra experiment after the scratch run failed.
        path = checkpoint_path(map_id, args.root)
        options={'episodes':args.map2_episodes,'seed':args.map2_seed} if map_id==2 else {}
        run(f"map{map_id}_final", path, map_id,**options)
        run(f"map{map_id}_stochastic", path, map_id, stochastic=True,**options)
        run(f"map{map_id}_canonical", path, map_id, canonical=True)
        run(f"map{map_id}_variants", path, map_id, variants=True)
        run(f"map{map_id}_untrained", path, map_id, policy="untrained")
        run(f"map{map_id}_random", path, map_id, policy="random")
    for map_id in [0, 3]:
        for variant, folder in [("base", f"ppo_v2_base_map{map_id}_seed0"),
                                ("shaped500", f"ppo_v2_map{map_id}_seed0")]:
            run(f"map{map_id}_{variant}", args.root / folder / "checkpoint_0500.npz", map_id)
    for seed in [1, 2]:
        run(f"map3_shaped_seed{seed}", args.root / f"ppo_v2_map3_seed{seed}/checkpoint_0500.npz", 3)
    run("map1_scratch", args.root / "ppo_v2_map1_seed0/checkpoint_1200.npz", 1)
    for map_id in [0, 1, 2]:
        run(f"zero_shot_map{map_id}", args.root / "ppo_v2_map3_seed0/checkpoint_1000.npz", map_id)
    run('map2_before_safety_fresh',args.root / 'ppo_v2_map2_seed0/checkpoint_1600.npz',2,
        episodes=args.map2_episodes,seed=args.map2_seed)
    (out / "summary.json").write_text(json.dumps({"protocol": "frozen checkpoints; test resets not used for selection",
        "seed_start": args.seed, "episodes_randomized_per_case": args.episodes, "results": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
