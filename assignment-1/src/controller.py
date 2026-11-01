"""Privileged demo driver: pure pursuit plus nine short safety forecasts.

The driver has full state. It is not a learned policy and does not establish
what a future policy can achieve from observations alone.
"""
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
from env.robotaxi import move_obstacles, to_local, vehicle_motion, swept_clearance


def tracking(position, heading, speed, goal, path, p):
    distances = jnp.linalg.norm(path - position, axis=1)
    nearest = jnp.argmin(distances)
    eligible = (jnp.arange(path.shape[0]) >= nearest) & (distances >= 2.1 + .35 * speed)
    index = jnp.where(jnp.any(eligible), jnp.argmax(eligible), path.shape[0] - 1)
    local = to_local(path[index] - position, heading)
    alpha = jnp.arctan2(local[1], local[0])
    steer = jnp.arctan2(2 * p.wheelbase * jnp.sin(alpha), jnp.maximum(jnp.linalg.norm(local), .5))
    steer = jnp.where(jnp.abs(alpha) > 1.2, jnp.sign(alpha) * p.max_steer, steer)
    distance = jnp.linalg.norm(goal - position)
    desired_speed = jnp.minimum(3., jnp.sqrt(3. * jnp.maximum(distance - .3 * p.goal_radius, 0.)))
    return steer, desired_speed / (1 + 2 * jnp.abs(alpha)), distance


def candidate_action(position, heading, speed, state, params, speed_factor, steer_bias):
    p = params
    steer, desired, distance = tracking(position, heading, speed, state.goal_pos, state.path_array, p)
    required = 2.5 * (desired * speed_factor - speed) + p.drag * speed + p.rolling_friction
    throttle = required / jnp.where(required >= 0, p.max_accel, p.max_brake)
    throttle = jnp.where((distance <= .8 * p.goal_radius) | (speed_factor == 0), -1., throttle)
    return jnp.clip(jnp.array([throttle, (steer + steer_bias) / p.max_steer]), -1., 1.)


@partial(jax.jit, static_argnames=("params",))
def demo_action(state, params):
    p, s = params, state
    def traffic_tick(carry, _):
        old, velocity = carry
        traffic_state = eqx.tree_at(lambda x: (x.kinematic_obstacles, x.kinematic_obst_velocities),
                                    s, (old, velocity))
        new, velocity = move_obstacles(traffic_state, p)
        return (new, velocity), (old, new)
    _, (traffic_before, traffic_after) = jax.lax.scan(
        traffic_tick, (s.kinematic_obstacles, s.kinematic_obst_velocities), None, length=54)
    arc = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.linalg.norm(jnp.diff(s.path_array, axis=0), axis=1))])

    def route_progress(position):
        index = jnp.argmin(jnp.linalg.norm(s.path_array - position, axis=1))
        following = jnp.minimum(index + 1, s.path_array.shape[0] - 1)
        direction = s.path_array[following] - s.path_array[index]
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-6)
        return arc[index] + jnp.dot(position - s.path_array[index], direction)
    initial_progress = route_progress(s.agent_pos)

    def alignment(position, heading):
        index = jnp.argmin(jnp.linalg.norm(s.path_array - position, axis=1))
        target = s.path_array[jnp.minimum(index + 1, s.path_array.shape[0] - 1)] - position
        direction = target / jnp.maximum(jnp.linalg.norm(target), 1e-6)
        return jnp.dot(direction, jnp.array([jnp.cos(heading), jnp.sin(heading)]))
    initial_alignment = alignment(s.agent_pos, s.heading)

    def assess(speed_factor, steer_bias):
        def tick(carry, traffic):
            position, heading, speed, clearance = carry
            before, after = traffic
            action = candidate_action(position, heading, speed, s, p, speed_factor, steer_bias)
            new_pos, heading, speed = vehicle_motion(position, heading, speed, action, p)
            gap = swept_clearance(position, new_pos,
                                  jnp.concatenate([s.static_obstacles, before]),
                                  jnp.concatenate([s.static_obstacles, after]), p.radius)
            boundary = jnp.minimum(new_pos - p.radius,
                                   (jnp.array(p.map_height_width[::-1]) - 1) * p.cell_size - p.radius - new_pos)
            gap = jnp.minimum(gap, jnp.min(boundary))
            return (new_pos, heading, speed, jnp.minimum(clearance, gap)), None
        end, _ = jax.lax.scan(tick, (s.agent_pos, s.heading, s.speed, jnp.array(jnp.inf)),
                              (traffic_before, traffic_after))
        position, heading, _, clearance = end
        deviation = jnp.min(jnp.linalg.norm(s.path_array - position, axis=1))
        progress = route_progress(position) - initial_progress
        score = (progress - .25 * deviation - .1 * jnp.abs(steer_bias)
                 + 1.5 * (alignment(position, heading) - initial_alignment)
                 + .1 * jnp.minimum(clearance, 2.))
        # A stopped car facing across the route must first turn while moving.
        # Prefer a safe moving candidate over standing still far from the goal.
        score -= jnp.where((speed_factor == 0) & (s.speed < .1)
                            & (jnp.linalg.norm(s.goal_pos - s.agent_pos) > p.goal_radius), .5, 0.)
        return clearance, score

    speed_factors = jnp.repeat(jnp.array([1., .55, 0.]), 3)
    steer_biases = jnp.tile(jnp.array([0., -.25, .25]), 3)
    gaps, scores = jax.vmap(assess)(speed_factors, steer_biases)
    safe = gaps >= .3
    ranks = jnp.where(jnp.any(safe), jnp.where(safe, scores, -jnp.inf), gaps + .01 * scores)
    index = jnp.argmax(ranks)
    return candidate_action(s.agent_pos, s.heading, s.speed, s, p,
                             speed_factors[index], steer_biases[index])
