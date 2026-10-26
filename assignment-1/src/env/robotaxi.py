"""JAX robotaxi environment. Lengths are world units; angles are radians.

The supplied BaseEnv source stays unchanged; its ray casting helpers are reused.
"""
from functools import partial
import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from env.base import BaseEnv, BaseEnvParams, BaseEnvState
from env.tasks import BaseTaskSampler


class TaxiParams(BaseEnvParams):
    cell_size: float
    radius: float
    goal_radius: float
    wheelbase: float
    max_speed: float
    max_accel: float
    max_brake: float
    max_steer: float
    drag: float
    rolling_friction: float
    progress_weight: float
    goal_bonus: float
    collision_penalty: float
    time_cost: float


class TaxiState(BaseEnvState):
    heading: chex.Array
    speed: chex.Array
    terminated: chex.Array


class TaxiObservation(eqx.Module):
    # These three fields also support the supplied frontend.
    distance_to_path: chex.Array
    direction_of_path: chex.Array
    collision_rays: chex.Array
    goal_local: chex.Array
    path_local: chex.Array
    speed: chex.Array
    ray_distances: chex.Array

    def as_vector(self):
        return jnp.concatenate([
            jnp.atleast_1d(self.speed), self.goal_local, self.path_local,
            jnp.atleast_1d(self.distance_to_path), self.ray_distances,
        ]).astype(jnp.float32)


def wrap_angle(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def to_local(vector, heading):
    c, s = jnp.cos(heading), jnp.sin(heading)
    return jnp.array([c * vector[0] + s * vector[1],
                      -s * vector[0] + c * vector[1]])


def plan_path(agent_pos, goal_pos, static_obstacles, p):
    """Four-neighbour Bellman planner, without grid wraparound.

    The reference path sees static obstacles only. Moving objects remain in
    perception/collision detection rather than temporarily becoming grid walls.
    """
    h, w = p.map_height_width

    def cell(pos):
        xy = jnp.rint(pos / p.cell_size).astype(jnp.int32)
        return jnp.clip(jnp.array([h - 1 - xy[1], xy[0]]),
                        jnp.array([0, 0]), jnp.array([h - 1, w - 1]))

    indices = jax.vmap(cell)(static_obstacles)
    blocked = jnp.zeros((h, w), dtype=jnp.bool_)
    blocked = blocked.at[indices[:, 0], indices[:, 1]].set(True)
    goal = cell(goal_pos)
    cost = jnp.full((h, w), jnp.inf).at[goal[0], goal[1]].set(0.)

    def neighbors(a):
        padded = jnp.pad(a, 1, constant_values=jnp.inf)
        return jnp.stack([padded[2:, 1:-1], padded[:-2, 1:-1],
                          padded[1:-1, 2:], padded[1:-1, :-2]])

    def relax(_, value):
        best = jnp.minimum(value, 1 + neighbors(value).min(axis=0))
        return jnp.where(blocked, jnp.inf, best)

    cost = jax.lax.fori_loop(0, h * w, relax, cost)
    moves = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])

    def trace(pos, _):
        candidates = jnp.clip(pos + moves, jnp.array([0, 0]), jnp.array([h - 1, w - 1]))
        values = cost[candidates[:, 0], candidates[:, 1]]
        next_pos = jax.lax.cond(
            jnp.isfinite(cost[pos[0], pos[1]]) & (cost[pos[0], pos[1]] > 0),
            lambda: candidates[jnp.argmin(values)], lambda: pos,
        )
        return next_pos, next_pos

    start = cell(agent_pos)
    _, cells = jax.lax.scan(trace, start, None,
                             length=p.path_array_shape_dtype[0][0] - 1)
    cells = jnp.concatenate([start[None, :], cells], axis=0)
    return jnp.stack([cells[:, 1], h - 1 - cells[:, 0]], axis=1).astype(jnp.float32) * p.cell_size


def vehicle_motion(position, heading, speed, action, p):
    """Midpoint integration of a no-slip bicycle and longitudinal resistance."""
    action = jnp.clip(jnp.nan_to_num(action), -1., 1.)
    acceleration = jnp.where(action[0] >= 0, p.max_accel, p.max_brake) * action[0]
    steering = action[1] * p.max_steer
    resistance = p.drag * speed + p.rolling_friction * (speed > 0)
    new_speed = jnp.clip(speed + p.step_size * (acceleration - resistance), 0., p.max_speed)
    midpoint_speed = (speed + new_speed) / 2
    yaw_rate = midpoint_speed * jnp.tan(steering) / p.wheelbase
    midpoint_heading = heading + 0.5 * p.step_size * yaw_rate
    new_pos = position + p.step_size * midpoint_speed * jnp.array([
        jnp.cos(midpoint_heading), jnp.sin(midpoint_heading)])
    return new_pos, wrap_angle(heading + p.step_size * yaw_rate), new_speed


def swept_clearance(agent_before, agent_after, obstacles_before, obstacles_after, radius):
    """Closest distance of linearly moving circle centres within a step."""
    if obstacles_before.shape[0] == 0:
        return jnp.array(jnp.inf, dtype=jnp.float32)
    relative_start = agent_before - obstacles_before
    relative_delta = (agent_after - agent_before) - (obstacles_after - obstacles_before)
    t = jnp.clip(-jnp.sum(relative_start * relative_delta, axis=1) /
                 jnp.maximum(jnp.sum(relative_delta**2, axis=1), 1e-12), 0., 1.)
    distances = jnp.linalg.norm(relative_start + t[:, None] * relative_delta, axis=1)
    return jnp.min(distances) - 2 * radius


class RobotaxiEnv(BaseEnv):
    @staticmethod
    def init_params(key, map_id=0, max_steps=2400, path_length=100,
                    discretization_scale=1, perception_radius=12.,
                    num_ray_sensors=32, fov=math.pi, fps=30,
                    cell_size=3., radius=1.):
        del key
        if map_id not in (0, 1, 2, 3):
            raise ValueError("map_id must be 0, 1, 2 or 3 (open demonstration)")
        if discretization_scale != 1:
            raise ValueError("Only discretization_scale=1 is supported")
        if fps <= 0 or max_steps < 1 or path_length < 2 or num_ray_sensors < 4:
            raise ValueError("Invalid episode, time step, path or sensor dimensions")
        if radius <= 0 or cell_size <= 2 * radius or perception_radius <= radius:
            raise ValueError("cell_size must exceed the collision diameter")
        task = BaseTaskSampler(min(map_id, 2))
        if map_id == 3:
            task.static_obstacles = jnp.array([
                [r, c] for r in range(10) for c in range(20)
                if r in (0, 9) or c in (0, 19)], dtype=jnp.float32)
            task.kinematic_obstacles = jnp.array([[3., 10.]])
            task.kinematic_obst_velocities = jnp.array([[0., -0.5]])
            task.agent_pos = jnp.array([5., 3.])
            task.goal_pos = jnp.array([5., 16.])
            task.agent_forward_dir = jnp.array([1., 0.])
        p = TaxiParams(
            max_steps_in_episode=max_steps, fps=fps, step_size=1. / fps,
            map_height_width=(task.height, task.width),
            nav_grid_shape_dtype=((task.height, task.width), jnp.float32),
            path_array_shape_dtype=((path_length, 2), jnp.float32),
            discretization_scale=1, num_ray_sensors=num_ray_sensors,
            agent_FOV=fov, perception_radius=perception_radius,
            cell_size=cell_size, radius=radius, goal_radius=radius,
            wheelbase=1.5, max_speed=5., max_accel=3., max_brake=5.,
            max_steer=0.6, drag=0.18, rolling_friction=0.12,
            progress_weight=1., goal_bonus=100., collision_penalty=100., time_cost=0.1,
        )

        def world(coords):
            return jnp.stack([coords[..., 1], task.height - 1 - coords[..., 0]], axis=-1).astype(jnp.float32) * cell_size

        pos, goal = world(task.agent_pos), world(task.goal_pos)
        obstacles = world(task.static_obstacles)
        heading = jnp.arctan2(task.agent_forward_dir[1], task.agent_forward_dir[0])
        state = TaxiState(
            time=jnp.asarray(0, dtype=jnp.int32), goal_pos=goal, agent_pos=pos,
            agent_forward_dir=task.agent_forward_dir.astype(jnp.float32),
            static_obstacles=obstacles, kinematic_obstacles=world(task.kinematic_obstacles),
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(jnp.float32),
            path_array=plan_path(pos, goal, obstacles, p), heading=heading,
            speed=jnp.array(0., dtype=jnp.float32), terminated=jnp.array(False),
        )
        return p, state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(key, env_params, init_state):
        del key
        return RobotaxiEnv.get_observation(init_state, env_params), init_state

    @staticmethod
    def get_observation(env_state, env_params):
        s, p = env_state, env_params
        distances = jnp.linalg.norm(s.path_array - s.agent_pos, axis=1)
        nearest = jnp.argmin(distances)
        next_idx = jnp.minimum(nearest + 1, s.path_array.shape[0] - 1)
        direction = s.path_array[next_idx] - s.agent_pos
        direction = jnp.where(jnp.linalg.norm(direction) > 1e-5,
                              direction, s.goal_pos - s.agent_pos)
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-6)
        anchors = s.path_array[:-1]
        segments = s.path_array[1:] - anchors
        fractions = jnp.clip(jnp.sum((s.agent_pos - anchors) * segments, axis=1) /
                             jnp.maximum(jnp.sum(segments**2, axis=1), 1e-8), 0., 1.)
        projections = anchors + fractions[:, None] * segments
        path_distance = jnp.min(jnp.linalg.norm(s.agent_pos - projections, axis=1))
        obstacles = jnp.concatenate([s.static_obstacles, s.kinematic_obstacles])
        rays = BaseEnv._generate_rays(s.agent_forward_dir, p.num_ray_sensors, p.agent_FOV)
        if obstacles.shape[0]:
            lengths = BaseEnv._ray_circle_intersections(s.agent_pos, rays, obstacles, p.radius)
            lengths = jnp.minimum(lengths, p.perception_radius)
        else:
            lengths = jnp.full((p.num_ray_sensors,), p.perception_radius)
        map_diagonal = p.cell_size * math.hypot(*p.map_height_width)
        return TaxiObservation(
            distance_to_path=path_distance / p.cell_size,
            direction_of_path=direction, collision_rays=rays * lengths[:, None],
            goal_local=to_local(s.goal_pos - s.agent_pos, s.heading) / map_diagonal,
            path_local=to_local(direction, s.heading), speed=s.speed / p.max_speed,
            ray_distances=lengths / p.perception_radius,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        del key
        s, p = env_state, env_params
        position, heading, speed = vehicle_motion(s.agent_pos, s.heading, s.speed, action, p)
        low = jnp.array([p.radius, p.radius])
        high = (jnp.array(p.map_height_width[::-1]) - 1) * p.cell_size - p.radius
        tentative = s.kinematic_obstacles + p.step_size * s.kinematic_obst_velocities
        reflected = jnp.where(tentative < low, 2 * low - tentative,
                             jnp.where(tentative > high, 2 * high - tentative, tentative))
        velocities = jnp.where((tentative < low) | (tentative > high),
                                -s.kinematic_obst_velocities, s.kinematic_obst_velocities)
        before = jnp.concatenate([s.static_obstacles, s.kinematic_obstacles])
        after = jnp.concatenate([s.static_obstacles, reflected])
        clearance = swept_clearance(s.agent_pos, position, before, after, p.radius)
        outside = jnp.any((position < low) | (position > high))
        collision = (clearance <= 0) | outside
        reached = (jnp.linalg.norm(position - s.goal_pos) <= p.goal_radius) & ~collision
        time = s.time + 1
        truncated = (time >= p.max_steps_in_episode) & ~reached & ~collision
        done = collision | reached | truncated
        progress = jnp.linalg.norm(s.agent_pos - s.goal_pos) - jnp.linalg.norm(position - s.goal_pos)
        reward = (p.progress_weight * progress - p.time_cost * p.step_size
                  + p.goal_bonus * reached - p.collision_penalty * collision)
        path = jax.lax.cond(
            time % p.fps == 0,
            lambda: plan_path(position, s.goal_pos, s.static_obstacles, p),
            lambda: s.path_array,
        )
        new_state = eqx.tree_at(
            lambda x: (x.time, x.agent_pos, x.agent_forward_dir, x.heading, x.speed,
                       x.kinematic_obstacles, x.kinematic_obst_velocities,
                       x.path_array, x.terminated), s,
            (time, position, jnp.array([jnp.cos(heading), jnp.sin(heading)]), heading,
             speed, reflected, velocities, path, done),
        )
        new_state = jax.tree.map(lambda old, new: jnp.where(s.terminated, old, new), s, new_state)
        reward = jnp.where(s.terminated, 0., reward)
        obs = RobotaxiEnv.get_observation(new_state, p)
        info = {
            "time": new_state.time, "speed": new_state.speed,
            "distance_to_goal": jnp.linalg.norm(new_state.agent_pos - s.goal_pos),
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "goal_reached": reached & ~s.terminated,
            "collision": collision & ~s.terminated,
            "truncated": truncated & ~s.terminated,
            "clearance": jnp.where(s.terminated, jnp.inf, clearance),
        }
        return obs, new_state, reward, new_state.terminated, info
