"""Export actual environment rollouts through the same renderer as manual play."""
import argparse
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
from evaluate import rollout
from utils.taxi_renderer import TaxiFrontend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", nargs="+", type=int, default=[0, 2, 3])
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo.mp4"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with imageio.get_writer(str(args.output), fps=30, codec="libx264", quality=7) as writer:
        for map_id in args.maps:
            params, initial = RobotaxiEnv.init_params(jax.random.PRNGKey(0), map_id=map_id)
            frontend = TaxiFrontend(params, initial, headless=True)
            initial_distance = float(np.linalg.norm(initial.agent_pos - initial.goal_pos))
            frontend.draw(map_id, "Heuristic controller / no training")
            for _ in range(30):
                writer.append_data(frontend.frame())

            def on_frame(state, obs, info, p, current_map):
                frontend.state, frontend.obs, frontend.info = state, obs, info
                episode_return = (initial_distance - float(info["distance_to_goal"])
                                  - int(state.time) * p.step_size * p.time_cost
                                  + p.goal_bonus * bool(info["goal_reached"])
                                  - p.collision_penalty * bool(info["collision"]))
                frontend.draw(current_map, "Heuristic controller / no training", episode_return)
                writer.append_data(frontend.frame())
                if int(state.time) == 300:
                    imageio.imwrite(args.output.parent / f"preview_map{current_map}.png", frontend.frame())

            result, _, _ = rollout(map_id, on_frame=on_frame)
            results.append(result)
            imageio.imwrite(args.output.parent / f"final_map{map_id}.png", frontend.frame())
            for _ in range(60):
                writer.append_data(frontend.frame())
            pygame.quit()
            print(json.dumps(result), flush=True)
    args.output.with_suffix(".json").write_text(json.dumps({"maps": results, "fps": 30}, indent=2) + "\n")


if __name__ == "__main__":
    main()
