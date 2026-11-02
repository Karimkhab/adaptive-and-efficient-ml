"""Reproduce the complete experiment matrix into a NEW directory."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/reproduced"))
    args = parser.parse_args()
    def run(folder, map_id, updates, seed=0, shaped=True, init=None,extra=None):
        command = [sys.executable, "src/train.py", "--output", str(args.root / folder),
                   "--map", str(map_id), "--updates", str(updates), "--seed", str(seed), "--randomize"]
        if shaped:
            command += ["--route-reward", "--parking-reward"]
        if init:
            command += ["--init-checkpoint", str(init)]
        if extra:
            command+=extra
        subprocess.run(command, check=True)
    run("ppo_v2_map0_seed0", 0, 1500)
    run("ppo_v2_map1_seed0", 1, 1200)
    run("ppo_v2_map2_seed0", 2, 1600)
    run("ppo_v2_map3_seed0", 3, 1000)
    run("ppo_v2_map1_transfer_seed0", 1, 1000,
        init=args.root / "ppo_v2_map0_seed0/checkpoint_0500.npz")
    for map_id in [0, 3]:
        run(f"ppo_v2_base_map{map_id}_seed0", map_id, 500, shaped=False)
    for seed in [1, 2]:
        run(f"ppo_v2_map3_seed{seed}", 3, 500, seed=seed)
    for name,seed,cruise,speed_cost,clearance_cost,margin in [('a',0,2.5,1,4,1),('b',1,3,.75,6,1.25)]:
        run(f'ppo_v2_map2_safe_{name}',2,800,seed=seed,
            init=args.root/'ppo_v2_map2_seed0/checkpoint_1600.npz',
            extra=['--learning-rate','0.0001','--entropy','0','--cruise-speed',str(cruise),
                   '--speed-cost',str(speed_cost),'--clearance-cost',str(clearance_cost),
                   '--clearance-margin',str(margin),'--clearance-speed'])
    from export_final import export
    export(args.root)


if __name__ == "__main__":
    main()
