"""A transparent path-following demonstration controller, not an RL policy.

It uses the privileged reference path and obstacle velocities. It is deliberately
separate from the observation contract exposed to a future learned agent.
"""
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
from env.robotaxi import to_local, move_obstacles, vehicle_motion


@partial(jax.jit, static_argnames=("params",))
def demo_action(state, params):
    p, s = params, state
    path = s.path_array
    # Follow a point a short distance ahead on the polyline, skipping repeats.
    nearest = jnp.argmin(jnp.linalg.norm(path - s.agent_pos, axis=1))
    lookahead = 2.1 + 0.35 * s.speed
    distances = jnp.linalg.norm(path - s.agent_pos, axis=1)
    eligible = (jnp.arange(path.shape[0]) >= nearest) & (distances >= lookahead)
    idx = jnp.where(jnp.any(eligible), jnp.argmax(eligible), path.shape[0] - 1)
    local = to_local(path[idx] - s.agent_pos, s.heading)
    alpha = jnp.arctan2(local[1], local[0])
    length = jnp.maximum(jnp.linalg.norm(local), 0.5)
    steering = jnp.arctan2(2 * p.wheelbase * jnp.sin(alpha), length)
    steering = jnp.where(jnp.abs(alpha) > 1.2, jnp.sign(alpha) * p.max_steer, steering)
    goal_distance = jnp.linalg.norm(s.goal_pos - s.agent_pos)
    desired_speed = jnp.minimum(3., jnp.sqrt(2. * p.max_brake * goal_distance))
    desired_speed = desired_speed / (1. + 2. * jnp.abs(alpha))

    # Predict close approaches over a short horizon at the current velocity.
    offsets = s.static_obstacles - s.agent_pos
    relative_velocity = jnp.broadcast_to(-s.speed * s.agent_forward_dir, offsets.shape)
    t = jnp.clip(-jnp.sum(offsets * relative_velocity, axis=1) /
                 jnp.maximum(jnp.sum(relative_velocity**2, axis=1), 1e-6), 0., 1.3)
    closest = jnp.linalg.norm(offsets + t[:, None] * relative_velocity, axis=1)
    ahead = offsets @ s.agent_forward_dir > 0
    static_danger = jnp.any(ahead & (closest < 2 * p.radius + .25))

    # Forecast the scripted traffic with the same wall-bounce rule as the world.
    # A constant-velocity forecast is unsafe immediately before an obstacle bounces.
    def predict(carry, _):
        pos, angle, speed, traffic, velocity = carry
        traffic_state = eqx.tree_at(lambda x: (x.kinematic_obstacles, x.kinematic_obst_velocities),
                                    s, (traffic, velocity))
        next_traffic, next_velocity = move_obstacles(traffic_state, p)
        acceleration = 2.5 * (desired_speed - speed) + p.drag * speed + p.rolling_friction
        throttle = acceleration / jnp.where(acceleration >= 0, p.max_accel, p.max_brake)
        pos, angle, speed = vehicle_motion(pos, angle, speed,
                                           jnp.array([throttle, steering / p.max_steer]), p)
        clearance = jnp.min(jnp.linalg.norm(next_traffic - pos, axis=1))
        return (pos, angle, speed, next_traffic, next_velocity), clearance

    if s.kinematic_obstacles.shape[0]:
        _, forecast = jax.lax.scan(predict, (s.agent_pos, s.heading, s.speed,
                                           s.kinematic_obstacles, s.kinematic_obst_velocities),
                                  None, length=54)
        traffic_danger = jnp.min(forecast) < 2 * p.radius + .2
    else:
        traffic_danger = False
    danger = static_danger | traffic_danger
    desired_speed = jnp.where(danger, 0., desired_speed)
    required_accel = 2.5 * (desired_speed - s.speed) + p.drag * s.speed + p.rolling_friction
    throttle = required_accel / jnp.where(required_accel >= 0, p.max_accel, p.max_brake)
    return jnp.clip(jnp.array([throttle, steering / p.max_steer]), -1., 1.)
