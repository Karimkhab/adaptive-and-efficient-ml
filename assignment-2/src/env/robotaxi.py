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


DEFAULT_HEADING_NOISE = math.pi / 18


class TaxiParams(BaseEnvParams):
    cell_size: float
    radius: float
    goal_radius: float
    goal_speed: float
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
    truncated: chex.Array

    @property
    def done(self):
        return self.terminated | self.truncated


class TaxiObservation(eqx.Module):
    # These three fields also support the supplied frontend.
    distance_to_path: chex.Array
    direction_of_path: chex.Array
    collision_rays: chex.Array
    goal_local: chex.Array
    path_local: chex.Array
    speed: chex.Array
    ray_distances: chex.Array
    ray_closing_speeds: chex.Array

    def as_vector(self):
        return jnp.concatenate([
            self.speed[..., None], self.goal_local, self.path_local,
            self.distance_to_path[..., None], self.ray_distances,
            self.ray_closing_speeds,
        ], axis=-1).astype(jnp.float32)


def wrap_angle(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def to_local(vector, heading):
    c, s = jnp.cos(heading), jnp.sin(heading)
    return jnp.array([c * vector[0] + s * vector[1],
                      -s * vector[0] + c * vector[1]])


def map_extent(p):
    return (jnp.array(p.map_height_width[::-1]) - 1) * p.cell_size


def boundary_clearance(position, p):
    return jnp.minimum(position, map_extent(p) - position).min(axis=-1) - p.radius


def plan_path(agent_pos, goal_pos, static_obstacles, p):
    """Four-neighbour Bellman planner, without grid wraparound.

    The reference path sees static obstacles only. Moving objects remain in
    perception/collision detection rather than temporarily becoming grid walls.
    """
    h, w = p.map_height_width

    def cell(pos):
        xy = jnp.rint(pos / p.cell_size).astype(jnp.int32)
        return jnp.clip(jnp.array([h - 1 - xy[1], xy[0]]),
                        jnp.array([0, 0], dtype=xy.dtype),
                        jnp.array([h - 1, w - 1], dtype=xy.dtype))

    indices = jax.vmap(cell)(static_obstacles)
    blocked = jnp.ones((h, w), dtype=jnp.bool_).at[1:-1, 1:-1].set(False)
    blocked = blocked.at[indices[:, 0], indices[:, 1]].set(True)
    goal = cell(goal_pos)
    cost = jnp.full((h, w), jnp.inf, dtype=agent_pos.dtype).at[goal[0], goal[1]].set(0.)

    def neighbors(a):
        padded = jnp.pad(a, 1, constant_values=jnp.inf)
        return jnp.stack([padded[2:, 1:-1], padded[:-2, 1:-1],
                          padded[1:-1, 2:], padded[1:-1, :-2]])

    def relax(_, value):
        best = jnp.minimum(value, 1 + neighbors(value).min(axis=0))
        return jnp.where(blocked, jnp.inf, best)

    cost = jax.lax.fori_loop(0, h * w, relax, cost)
    moves = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]], dtype=jnp.int32)

    def trace(pos, _):
        candidates = jnp.clip(pos + moves, jnp.array([0, 0], dtype=pos.dtype),
                              jnp.array([h - 1, w - 1], dtype=pos.dtype))
        values = cost[candidates[:, 0], candidates[:, 1]]
        next_pos = jax.lax.cond(
            jnp.isfinite(cost[pos[0], pos[1]]) & (cost[pos[0], pos[1]] > 0),
            lambda: candidates[jnp.argmin(values)], lambda: pos,
        )
        return next_pos, next_pos

    def centers(cells):
        return jnp.stack([cells[..., 1], h - 1 - cells[..., 0]], axis=-1).astype(agent_pos.dtype) * p.cell_size

    rounded = cell(agent_pos)
    offsets = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                         [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=jnp.int32)
    candidates = jnp.clip(rounded + offsets, jnp.array([0, 0], dtype=rounded.dtype),
                          jnp.array([h - 1, w - 1], dtype=rounded.dtype))
    entries = centers(candidates)
    entry_clearance = jax.vmap(lambda end: swept_clearance(
        agent_pos, end, static_obstacles, static_obstacles, p.radius))(entries)
    safe = ((entry_clearance > 0) & (boundary_clearance(entries, p) > 0)
            & jnp.isfinite(cost[candidates[:, 0], candidates[:, 1]]))
    index = jnp.argmin(jnp.where(safe, jnp.linalg.norm(entries - agent_pos, axis=1), jnp.inf))
    start = candidates[index]
    _, cells = jax.lax.scan(trace, start, None,
                            length=p.path_array_shape_dtype[0][0] - 1)
    path = centers(jnp.concatenate([start[None, :], cells], axis=0))
    path = jnp.where(index != 0, jnp.concatenate([agent_pos[None, :], path[:-1]]), path)
    return jnp.where(jnp.any(safe), path, jnp.broadcast_to(agent_pos, path.shape))


def vehicle_motion(position, heading, speed, action, p):
    """Integrate bounded longitudinal speed and follow the bicycle heading."""
    action = jnp.clip(jnp.nan_to_num(action), -1., 1.)
    acceleration = jnp.where(action[0] >= 0, p.max_accel, p.max_brake) * action[0]
    steering = action[1] * p.max_steer
    net_accel = acceleration - p.drag * speed - p.rolling_friction
    free_speed = speed + p.step_size * net_accel
    new_speed = jnp.clip(free_speed, 0., p.max_speed)
    # Acceleration ends early if the vehicle stops or hits the speed cap.
    limit_time = jnp.abs(new_speed - speed) / jnp.maximum(jnp.abs(net_accel), 1e-12)
    changing_time = jnp.where(free_speed == new_speed, p.step_size, limit_time)
    travel = (speed + new_speed) * .5 * changing_time + new_speed * (p.step_size - changing_time)
    yaw = travel * jnp.tan(steering) / p.wheelbase
    midpoint_heading = heading + .5 * yaw
    new_pos = position + travel * jnp.array([jnp.cos(midpoint_heading), jnp.sin(midpoint_heading)])
    return new_pos, wrap_angle(heading + yaw), new_speed


def swept_clearance(agent_before, agent_after, obstacles_before, obstacles_after, radius):
    """Closest distance of linearly moving circle centres within a step."""
    if obstacles_before.shape[0] == 0:
        return jnp.array(jnp.inf, dtype=agent_before.dtype)
    relative_start = agent_before - obstacles_before
    relative_delta = (agent_after - agent_before) - (obstacles_after - obstacles_before)
    t = jnp.clip(-jnp.sum(relative_start * relative_delta, axis=1) /
                 jnp.maximum(jnp.sum(relative_delta**2, axis=1), 1e-12), 0., 1.)
    distances = jnp.linalg.norm(relative_start + t[:, None] * relative_delta, axis=1)
    return jnp.min(distances) - 2 * radius


def move_obstacles(state, params):
    """Scripted traffic: axis-aligned groups reverse before walls/each other.

    Objects travelling along the same axis reverse together, preserving spacing
    in the supplied h/v groups. This is kinematic traffic, not a rigid-body solver.
    A conservative hold handles a step where neither direction is clear.
    """
    p = params
    positions, velocities = state.kinematic_obstacles, state.kinematic_obst_velocities
    if positions.shape[0] == 0:
        return positions, velocities

    def unsafe(candidate):
        walls = jax.vmap(lambda a, b: swept_clearance(
            a, b, state.static_obstacles, state.static_obstacles, p.radius
        ))(positions, candidate) <= 1e-5
        outside = boundary_clearance(candidate, p) <= 0
        relative = positions[:, None, :] - positions[None, :, :]
        delta = (candidate - positions)[:, None, :] - (candidate - positions)[None, :, :]
        t = jnp.clip(-jnp.sum(relative * delta, axis=-1) /
                     jnp.maximum(jnp.sum(delta * delta, axis=-1), 1e-12), 0., 1.)
        distance = jnp.linalg.norm(relative + t[..., None] * delta, axis=-1)
        contact = (distance <= 2 * p.radius + 1e-5) & ~jnp.eye(positions.shape[0], dtype=jnp.bool_)
        return walls | outside | jnp.any(contact, axis=1)

    tentative = positions + p.step_size * velocities
    hazard = unsafe(tentative)
    axes = jnp.argmax(jnp.abs(velocities), axis=1)
    same_group = axes[:, None] == axes[None, :]
    reverse = jnp.any(same_group & hazard[None, :], axis=1)
    velocities = jnp.where(reverse[:, None], -velocities, velocities)
    candidate = positions + p.step_size * velocities
    # No penetration is preferable to teleporting through a narrow passage.
    hold = jnp.any(unsafe(candidate))
    return jnp.where(hold, positions, candidate), velocities


class RobotaxiEnv(BaseEnv):
    OBSERVATION_VERSION = 2
    ENVIRONMENT_REVISION = "consistent_geometry_v1"
    ACTION_SHAPE = (2,)

    @staticmethod
    def observation_size(env_params):
        return 6 + 2 * env_params.num_ray_sensors

    @staticmethod
    def init_params(key, map_id=0, max_steps=3000, path_length=100,
                    discretization_scale=1, perception_radius=12.,
                    num_ray_sensors=32, fov=math.pi, fps=30,
                    cell_size=5., radius=1., *, cross_traffic=False):
        del key
        if map_id not in (0, 1, 2, 3, 4):
            raise ValueError("map_id must be between 0 and 4")
        if cross_traffic and map_id != 4:
            raise ValueError("cross_traffic is only a variant of map 4")
        if discretization_scale != 1:
            raise ValueError("Only discretization_scale=1 is supported")
        if fps <= 0 or max_steps < 1 or path_length < 2 or num_ray_sensors < 4:
            raise ValueError("Invalid episode, time step, path or sensor dimensions")
        if radius <= 0 or cell_size <= 2 * radius or perception_radius <= radius:
            raise ValueError("cell_size must exceed the collision diameter")
        physics_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        task = BaseTaskSampler(min(map_id, 2))
        if map_id in (3, 4):
            # The two map 4 variants share the layout and static reference path.
            static_cells = [
                [r, c] for r in range(10) for c in range(20)
                if r in (0, 9) or c in (0, 19)]
            if map_id == 4:
                static_cells.extend((r, c) for r in range(4, 8) for c in range(8, 12))
                # Close the gaps between perimeter colliders to the vehicle.
                static_cells.extend((r, c) for r in (4, 7) for c in (8.5, 9.5, 10.5))
                static_cells.extend((r, c) for r in (4.5, 5.5, 6.5) for c in (8, 11))
            task.static_obstacles = jnp.array(static_cells, dtype=physics_dtype)
            if map_id == 3:
                task.kinematic_obstacles = jnp.array([[3., 10.]])
                task.kinematic_obst_velocities = jnp.array([[0., -0.5]])
            elif cross_traffic:
                task.kinematic_obstacles = jnp.array([[1., 13.]])
                task.kinematic_obst_velocities = jnp.array([[0., -0.62]])
            else:
                task.kinematic_obstacles = jnp.empty((0, 2), dtype=physics_dtype)
                task.kinematic_obst_velocities = jnp.empty((0, 2), dtype=physics_dtype)
            task.agent_pos = jnp.array([5., 3.])
            task.goal_pos = jnp.array([5., 16.])
            task.agent_forward_dir = jnp.array([1., 0.])
        p = TaxiParams(
            max_steps_in_episode=max_steps, fps=fps, step_size=1. / fps,
            map_height_width=(task.height, task.width),
            nav_grid_shape_dtype=((task.height, task.width), physics_dtype),
            path_array_shape_dtype=((path_length, 2), physics_dtype),
            discretization_scale=1, num_ray_sensors=num_ray_sensors,
            agent_FOV=fov, perception_radius=perception_radius,
            cell_size=cell_size, radius=radius, goal_radius=radius, goal_speed=0.001,
            wheelbase=1.5, max_speed=5., max_accel=3., max_brake=5.,
            max_steer=0.6, drag=0.18, rolling_friction=0.12,
            progress_weight=1., goal_bonus=100., collision_penalty=100., time_cost=0.1,
        )

        def world(coords):
            return jnp.stack([coords[..., 1], task.height - 1 - coords[..., 0]], axis=-1).astype(physics_dtype) * cell_size

        pos, goal = world(task.agent_pos), world(task.goal_pos)
        obstacles = world(task.static_obstacles)
        forward = task.agent_forward_dir.astype(physics_dtype)
        heading = jnp.arctan2(forward[1], forward[0])
        state = TaxiState(
            time=jnp.asarray(0, dtype=jnp.int32), goal_pos=goal, agent_pos=pos,
            agent_forward_dir=forward,
            static_obstacles=obstacles, kinematic_obstacles=world(task.kinematic_obstacles),
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(physics_dtype),
            path_array=plan_path(pos, goal, obstacles, p), heading=heading,
            speed=jnp.array(0., dtype=physics_dtype), terminated=jnp.array(False),
            truncated=jnp.array(False),
        )
        return p, state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params", "randomize"))
    def reset(key, env_params, init_state, randomize=False, heading_noise=DEFAULT_HEADING_NOISE):
        if randomize:
            init_state = sample_initial_state(key, env_params, init_state, heading_noise)
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
        # Cast to the physical rectangle from the vehicle center, just like
        # the distances to circular obstacle surfaces.
        edge = jnp.where(rays > 0, map_extent(p), 0.)
        safe_direction = jnp.where(rays == 0, 1., rays)
        wall_hits = jnp.where(rays == 0, jnp.inf, (edge - s.agent_pos) / safe_direction).min(axis=1)
        wall_hits = jnp.maximum(wall_hits, 0.)
        obstacle_velocity = jnp.concatenate([
            jnp.zeros_like(s.static_obstacles), s.kinematic_obst_velocities])
        if obstacles.shape[0]:
            hits = jax.vmap(lambda center: BaseEnv._ray_circle_intersections(
                s.agent_pos, rays, center[None, :], p.radius
            ), out_axes=1)(obstacles)
            nearest_obstacle = jnp.argmin(hits, axis=1)
            circle_hits = jnp.min(hits, axis=1)
            hit_velocity = jnp.where((circle_hits < wall_hits)[:, None],
                                     obstacle_velocity[nearest_obstacle], 0.)
        else:
            circle_hits = jnp.full_like(wall_hits, jnp.inf)
            hit_velocity = jnp.zeros_like(rays)
        raw_lengths = jnp.minimum(circle_hits, wall_hits)
        lengths = jnp.minimum(raw_lengths, p.perception_radius)
        relative_velocity = hit_velocity - s.speed * s.agent_forward_dir
        closing = -jnp.sum(relative_velocity * rays, axis=1)
        speed_scale = p.max_speed + jnp.max(jnp.linalg.norm(obstacle_velocity, axis=1), initial=0.)
        closing = jnp.where(raw_lengths <= p.perception_radius, closing / speed_scale, 0.)
        map_diagonal = p.cell_size * math.hypot(*p.map_height_width)
        return TaxiObservation(
            distance_to_path=path_distance / p.cell_size,
            direction_of_path=direction, collision_rays=rays * lengths[:, None],
            goal_local=to_local(s.goal_pos - s.agent_pos, s.heading) / map_diagonal,
            path_local=to_local(direction, s.heading), speed=s.speed / p.max_speed,
            ray_distances=lengths / p.perception_radius,
            ray_closing_speeds=closing,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        del key
        s, p = env_state, env_params
        action = jnp.asarray(action, dtype=s.agent_pos.dtype)
        if action.shape != RobotaxiEnv.ACTION_SHAPE:
            raise ValueError("Action must have shape (2,): acceleration and steering")
        position, heading, speed = vehicle_motion(s.agent_pos, s.heading, s.speed, action, p)
        reflected, velocities = move_obstacles(s, p)
        before = jnp.concatenate([s.static_obstacles, s.kinematic_obstacles])
        after = jnp.concatenate([s.static_obstacles, reflected])
        clearance = swept_clearance(s.agent_pos, position, before, after, p.radius)
        wall_gap = jnp.minimum(boundary_clearance(s.agent_pos, p), boundary_clearance(position, p))
        clearance = jnp.minimum(clearance, wall_gap)
        collision = clearance <= 0
        reached = ((jnp.linalg.norm(position - s.goal_pos) <= p.goal_radius)
                   & (speed <= p.goal_speed) & ~collision)
        time = s.time + 1
        truncated = (time >= p.max_steps_in_episode) & ~reached & ~collision
        done = collision | reached | truncated
        progress = jnp.linalg.norm(s.agent_pos - s.goal_pos) - jnp.linalg.norm(position - s.goal_pos)
        reward = (p.progress_weight * progress - p.time_cost * p.step_size
                  + p.goal_bonus * reached - p.collision_penalty * collision)
        # Static geometry does not require snapping a new route every second.
        # Keep the guide stable near it; replan after a genuine route deviation.
        off_route = jnp.min(jnp.linalg.norm(s.path_array - position, axis=1)) > p.cell_size
        def replan():
            candidate = plan_path(position, s.goal_pos, s.static_obstacles, p)
            return jnp.where(jnp.all(candidate[-1] == s.goal_pos), candidate, s.path_array)

        path = jax.lax.cond((time % p.fps == 0) & off_route & ~done, replan, lambda: s.path_array)
        new_state = eqx.tree_at(
            lambda x: (x.time, x.agent_pos, x.agent_forward_dir, x.heading, x.speed,
                       x.kinematic_obstacles, x.kinematic_obst_velocities,
                       x.path_array, x.terminated, x.truncated), s,
            (time, position, jnp.array([jnp.cos(heading), jnp.sin(heading)]), heading,
             speed, reflected, velocities, path, reached | collision, truncated),
        )
        new_state = jax.tree.map(lambda old, new: jnp.where(s.done, old, new), s, new_state)
        reward = jnp.where(s.done, 0., reward)
        obs = RobotaxiEnv.get_observation(new_state, p)
        info = {
            "time": new_state.time, "speed": new_state.speed,
            "distance_to_goal": jnp.linalg.norm(new_state.agent_pos - s.goal_pos),
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "goal_reached": reached & ~s.done,
            "collision": collision & ~s.done,
            "terminated": new_state.terminated,
            "truncated": new_state.truncated,
            "clearance": jnp.where(s.done, jnp.inf, clearance),
            "final_observation": obs.as_vector(),
            "final_observation_valid": new_state.done,
        }
        return obs, new_state, reward, new_state.done, info


def sample_initial_state(key, params, initial, heading_noise=DEFAULT_HEADING_NOISE):
    """Optional reproducible training resets, with unchanged shapes and map.

    Heading defaults to +/-10 degrees, initial speed [0,.5], traffic speed [.8,1.2],
    and phase [0,1] second. Fixed evaluation reset remains the default.
    """
    angle_key, speed_key, traffic_key, phase_key = jax.random.split(key, 4)
    # Use the same random parameters for a seed in both precision modes.
    heading = initial.heading + jax.random.uniform(
        angle_key, (), dtype=jnp.float32, minval=-heading_noise, maxval=heading_noise)
    speed = jax.random.uniform(speed_key, (), dtype=jnp.float32, minval=0., maxval=.5)
    speed = speed.astype(initial.speed.dtype)
    scale = jax.random.uniform(traffic_key, (), dtype=jnp.float32, minval=.8, maxval=1.2)
    state = eqx.tree_at(lambda s: (s.heading, s.agent_forward_dir, s.speed, s.kinematic_obst_velocities),
                        initial, (heading, jnp.array([jnp.cos(heading), jnp.sin(heading)]),
                                  speed, initial.kinematic_obst_velocities * scale))
    phase = jax.random.randint(phase_key, (), 0, params.fps + 1, dtype=jnp.int32)
    def advance(_, state):
        positions, velocities = move_obstacles(state, params)
        return eqx.tree_at(lambda s: (s.kinematic_obstacles, s.kinematic_obst_velocities),
                            state, (positions, velocities))
    return jax.lax.fori_loop(0, phase, advance, state)
