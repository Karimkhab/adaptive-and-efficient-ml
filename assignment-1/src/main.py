"""Manual driving or a deterministic demo; video export works without a display."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument("--demo", action="store_true", help="Use a heuristic driver, not a trained policy")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", help="MP4 output path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2400)
    args = parser.parse_args()
    if args.headless and not args.demo:
        parser.error("--headless requires --demo")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    import jax
    from env.robotaxi import RobotaxiEnv
    from utils.taxi_renderer import TaxiFrontend
    params, state = RobotaxiEnv.init_params(jax.random.PRNGKey(args.seed), map_id=args.map, max_steps=args.max_steps)
    TaxiFrontend(params, state, args.headless).run(args.map, args.demo, args.record)


if __name__ == "__main__":
    main()
