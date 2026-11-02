"""Run a checkpoint with observation-only control, optionally in a live window."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--map", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--seed", type=int, default=20000)
    args = parser.parse_args()
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import jax
    import pygame
    from env.robotaxi import RobotaxiEnv
    from policy import Policy
    from ppo import control_repeat
    from utils.taxi_renderer import TaxiFrontend
    policy = Policy(args.checkpoint)
    config = policy.config
    map_id = config["map"] if args.map is None else args.map
    p, initial = RobotaxiEnv.init_params(jax.random.PRNGKey(0), map_id=map_id, max_steps=config["max_steps"])
    obs, state = RobotaxiEnv.reset(jax.random.PRNGKey(args.seed), p, initial, randomize=args.randomize)
    front = TaxiFrontend(p, state, headless=args.headless)
    ret = 0.
    for t in range(config["max_steps"]):
        if not args.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z:
                        front.follow_camera = not front.follow_camera
                    elif event.key == pygame.K_l:
                        front.show_rays = not front.show_rays
                    elif event.key == pygame.K_c:
                        front.show_colliders = not front.show_colliders
        if t % control_repeat(config) == 0:
            action = policy(obs.as_vector())
        obs, state, reward, done, info = RobotaxiEnv.step(jax.random.PRNGKey(0), state, action, p)
        ret += float(reward)
        front.state, front.obs, front.info = state, obs, dict(info, action=action)
        if not args.headless:
            front.draw(map_id, "PPO / trained neural actor", ret)
            front.clock.tick(p.fps)
        if bool(done):
            break
    print(json.dumps({"map": map_id, "goal_reached": bool(info["goal_reached"]),
        "collision": bool(info["collision"]), "truncated": bool(info["truncated"]),
        "final_speed": float(state.speed), "final_distance": float(info["distance_to_goal"]),
        "steps": int(state.time), "raw_return": ret}))
    pygame.quit()


if __name__ == "__main__":
    main()
