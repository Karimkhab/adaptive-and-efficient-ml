"""Record frozen neural actors through the Assignment 1 pixel renderer."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
import imageio.v2 as imageio
import jax
import numpy as np
import pygame

from env.robotaxi import RobotaxiEnv
from policy import Policy
from ppo import control_repeat
from utils.taxi_renderer import TaxiFrontend

FINAL_UPDATES = {0: 1500, 1: 1200, 2: 1600, 3: 1000}


def checkpoint_path(map_id, root=Path("artifacts")):
    manifest=root / 'final_models.json'
    if manifest.exists():
        return root / json.loads(manifest.read_text())['checkpoints'][str(map_id)]
    if map_id == 1:
        return root / "ppo_v2_map1_transfer_seed0/checkpoint_1000.npz"
    return root / f"ppo_v2_map{map_id}_seed0/checkpoint_{FINAL_UPDATES[map_id]:04}.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=int, nargs="+", default=[3, 0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo.mp4"))
    parser.add_argument("--speed", type=int, default=2)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.speed < 1:
        parser.error("speed must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with imageio.get_writer(str(args.output), fps=30, codec="libx264", quality=7) as writer:
        for map_id in args.maps:
            checkpoint = checkpoint_path(map_id, args.root)
            actor = Policy(checkpoint)
            config = actor.config
            p, state = RobotaxiEnv.init_params(jax.random.PRNGKey(0), map_id=map_id, max_steps=config["max_steps"])
            obs = RobotaxiEnv.get_observation(state, p)
            front = TaxiFrontend(p, state, headless=True)
            front.playback_speed = args.speed
            mode = f"PPO / map {map_id} / seed {config['seed']} / {config['completed_updates']} updates"
            ret, info = 0., {}
            for _ in range(30):
                front.draw(map_id, mode, ret)
                writer.append_data(front.frame())
            rows = []
            for t in range(config["max_steps"]):
                if t % control_repeat(config) == 0:
                    action = actor(obs.as_vector())
                obs, state, reward, done, info = RobotaxiEnv.step(jax.random.PRNGKey(0), state, action, p)
                ret += float(reward)
                rows.append([int(state.time), *map(float, state.agent_pos), float(state.heading),
                             float(state.speed), *map(float, action), float(reward), float(info['clearance']), ret])
                front.state, front.obs, front.info = state, obs, dict(info, action=action)
                if t % args.speed == 0 or bool(done):
                    front.draw(map_id, mode, ret)
                    writer.append_data(front.frame())
                if t == 180:
                    imageio.imwrite(args.output.parent / f"preview_map{map_id}.png", front.frame())
                if bool(done):
                    break
            front.draw(map_id, mode, ret)
            for _ in range(60):
                writer.append_data(front.frame())
            imageio.imwrite(args.output.parent / f"final_map{map_id}.png", front.frame())
            np.savetxt(args.output.parent / f"demo_map{map_id}.csv", rows, delimiter=",",
                       header="step,x,y,heading,speed,throttle,steering,reward,clearance,return", comments="")
            result = {"map": map_id, "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "goal_reached": bool(info["goal_reached"]), "collision": bool(info["collision"]),
                "truncated": bool(info["truncated"]), "physics_steps": int(state.time),
                "sim_seconds": int(state.time) / p.fps, "raw_return": ret,
                "final_distance": float(info["distance_to_goal"]), "final_speed": float(state.speed),
                "min_clearance": min(r[-2] for r in rows)}
            results.append(result)
            print(json.dumps(result), flush=True)
            pygame.quit()
    args.output.with_suffix(".json").write_text(json.dumps({"fps": 30, "playback_speed": args.speed,
        "reset": "canonical fixed start", "episodes": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
