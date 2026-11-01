"""Small, explicit regression matrix: 8 initial conditions on each of 4 maps."""
import argparse
import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env.robotaxi import RobotaxiEnv, move_obstacles
from evaluate import rollout


VARIANTS = (
    ("baseline", {}),
    ("heading_minus_10deg", {"heading": -math.pi / 18}),
    ("heading_plus_10deg", {"heading": math.pi / 18}),
    ("initial_speed_0.5", {"speed": .5}),
    ("traffic_speed_0.8", {"traffic_scale": .8}),
    ("traffic_speed_1.2", {"traffic_scale": 1.2}),
    ("traffic_phase_0.5s", {"phase_seconds": .5}),
    ("traffic_phase_1.0s", {"phase_seconds": 1.}),
)


def vary_initial_state(params, state, *, heading=0., speed=0., traffic_scale=1., phase_seconds=0.):
    angle = state.heading + heading
    state = eqx.tree_at(lambda s: (s.heading, s.agent_forward_dir, s.speed, s.kinematic_obst_velocities),
                        state, (angle, jnp.array([jnp.cos(angle), jnp.sin(angle)]),
                                jnp.array(speed, dtype=jnp.float32),
                                state.kinematic_obst_velocities * traffic_scale))
    def tick(_, state):
        positions, velocities = move_obstacles(state, params)
        return eqx.tree_at(lambda s: (s.kinematic_obstacles, s.kinematic_obst_velocities),
                            state, (positions, velocities))
    return jax.lax.fori_loop(0, round(phase_seconds * params.fps), tick, state)


def run_suite(output=Path("artifacts/scenarios.json")):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for map_id in range(4):
        params, initial = RobotaxiEnv.init_params(jax.random.PRNGKey(0), map_id=map_id)
        for name, settings in VARIANTS:
            state = vary_initial_state(params, initial, **settings)
            result, trace, _ = rollout(map_id, params=params, initial_state=state)
            result.update({"variant": name, "settings": settings})
            result["passed"] = bool(result["goal_reached"] and not result["collision"]
                                     and not result["truncated"] and np.isfinite(trace).all()
                                     and result["min_clearance"] > 0
                                     and result["final_speed"] <= params.goal_speed)
            results.append(result)
            print(f"map {map_id} {name}: {'PASS' if result['passed'] else 'FAIL'} "
                  f"({result['steps']} steps, speed={result['final_speed']:.4f}, "
                  f"clearance={result['min_clearance']:.4f})", flush=True)
            if not result["passed"]:
                np.savetxt(output.parent / f"failed_map{map_id}_{name}.csv", trace, delimiter=",")
    document = {"purpose": "deterministic development/regression cases, not an independent quality estimate",
                "observation_version": RobotaxiEnv.OBSERVATION_VERSION,
                "total": len(results), "passed": sum(r["passed"] for r in results), "results": results}
    output.write_text(json.dumps(document, indent=2) + "\n")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/scenarios.json"))
    args = parser.parse_args()
    result = run_suite(args.output)
    raise SystemExit(0 if result["passed"] == result["total"] else 1)
