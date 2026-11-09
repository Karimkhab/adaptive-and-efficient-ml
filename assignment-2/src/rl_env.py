"""Action repeat, randomized resets, and explicit terminal observations."""
import equinox as eqx
import jax
import jax.numpy as jnp

from env.robotaxi import DEFAULT_HEADING_NOISE, RobotaxiEnv


def route_potential(position, path):
    """Negative distance remaining along the fixed initial reference polyline.

    Include distance to the polyline so a departure from it is not free.
    This shaping is a training experiment, not the evaluation score.
    """
    anchors, segments = path[:-1], path[1:] - path[:-1]
    lengths = jnp.linalg.norm(segments, axis=1)
    fractions = jnp.clip(jnp.sum((position - anchors) * segments, axis=1) /
                         jnp.maximum(lengths**2, 1e-8), 0., 1.)
    projections = anchors + fractions[:, None] * segments
    distance = jnp.linalg.norm(position - projections, axis=1)
    suffix = jnp.cumsum(lengths[::-1])[::-1]
    remaining = suffix - fractions * lengths
    # Each projection gives a continuous cost. Penalize distant projections
    # enough to keep a nearby branch from looking like a free shortcut.
    min_length = jnp.min(jnp.where(lengths > 0, lengths, jnp.inf))
    detour_weight = jnp.maximum(4., 1. + suffix[0] / min_length)
    nearest_distance = jnp.min(distance)
    adjusted = remaining + detour_weight * (distance - nearest_distance)
    return -(jnp.min(adjusted) + nearest_distance)


def make_env(map_id=3, repeat=6, max_steps=3000, heading_noise=None, route_reward=False,
             randomize=False, parking_reward=False, cruise_speed=4., speed_cost=.1,
             clearance_cost=0., clearance_margin=1., clearance_speed=False, *, cross_traffic=False):
    if heading_noise is None:
        heading_noise = DEFAULT_HEADING_NOISE if randomize else 0.
    p, initial = RobotaxiEnv.init_params(jax.random.PRNGKey(0), map_id=map_id, max_steps=max_steps,
                                        cross_traffic=cross_traffic)

    def reset(key):
        if randomize:
            obs, state = RobotaxiEnv.reset(key, p, initial, randomize=True, heading_noise=heading_noise)
            return state, obs.as_vector()
        heading = initial.heading + jax.random.uniform(
            key, (), dtype=jnp.float32, minval=-heading_noise, maxval=heading_noise)
        state = eqx.tree_at(lambda s: (s.heading, s.agent_forward_dir), initial,
                            (heading, jnp.array([jnp.cos(heading), jnp.sin(heading)])))
        return state, RobotaxiEnv.get_observation(state, p).as_vector()

    def step(state, action):
        before = state
        def body(carry, _):
            state, reward, success, collision, timeout, clearance, steps = carry
            active = ~state.done
            _, state, r, done, info = RobotaxiEnv.step(jax.random.PRNGKey(0), state, action, p)
            return (state, reward + r, success | info["goal_reached"],
                    collision | info["collision"], timeout | info["truncated"],
                    jnp.minimum(clearance, info["clearance"]), steps + active), None

        (state, reward, success, collision, timeout, clearance, steps), _ = jax.lax.scan(
            body, (state, jnp.array(0., dtype=state.agent_pos.dtype), jnp.array(False),
                   jnp.array(False), jnp.array(False),
                   jnp.array(jnp.inf, dtype=state.agent_pos.dtype), jnp.array(0, jnp.int32)),
            None, length=repeat)
        obs = RobotaxiEnv.get_observation(state, p).as_vector()
        raw_reward = reward
        if route_reward:
            euclidean_progress = (jnp.linalg.norm(before.agent_pos - before.goal_pos)
                                  - jnp.linalg.norm(state.agent_pos - state.goal_pos))
            route_progress = route_potential(state.agent_pos, initial.path_array) - route_potential(before.agent_pos, initial.path_array)
            reward = raw_reward - euclidean_progress + route_progress
        if parking_reward:
            distance = jnp.linalg.norm(state.agent_pos - state.goal_pos)
            desired_speed = jnp.minimum(cruise_speed, jnp.sqrt(3. * jnp.maximum(distance - .3, 0.)))
            if clearance_speed:
                desired_speed = jnp.minimum(desired_speed, jnp.sqrt(4. * jnp.maximum(clearance - .5, 0.)))
            reward -= speed_cost * (state.speed - desired_speed)**2 * steps * p.step_size
        if clearance_cost:
            reward -= clearance_cost * jnp.maximum(clearance_margin - clearance, 0.)**2 * steps * p.step_size
        return state, obs, reward, state.done, {
            "success": success, "collision": collision, "timeout": timeout,
            "clearance": clearance, "physics_steps": steps,
            "distance": jnp.linalg.norm(state.agent_pos - state.goal_pos), "raw_reward": raw_reward}

    return p, initial, reset, step
