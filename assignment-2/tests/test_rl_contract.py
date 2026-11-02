"""Stop-at-goal, velocity-aware sensing and episode boundaries for future RL."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from env.robotaxi import RobotaxiEnv
from utils.autoreset import AutoResetWrapper

KEY = jax.random.PRNGKey(7)


def change(state, **fields):
    for name, value in fields.items():
        state = eqx.tree_at(lambda s: getattr(s, name), state, jnp.asarray(value))
    return state


def test_high_speed_goal_crossing_does_not_count_as_arrival():
    p, s = RobotaxiEnv.init_params(KEY, map_id=3)
    s = change(s, agent_pos=s.goal_pos, speed=3.)
    _, _, reward, done, info = RobotaxiEnv.step(KEY, s, jnp.zeros(2), p)
    assert not done and not info['goal_reached'] and float(reward) < 1


def test_arrival_requires_real_braking_and_stops_inside_goal():
    p, s = RobotaxiEnv.init_params(KEY, map_id=3)
    s = change(s, agent_pos=s.goal_pos - jnp.array([.3, 0.]), speed=1.)
    positions = []
    for i in range(10):
        _, s, _, done, info = RobotaxiEnv.step(KEY, s, jnp.array([-1., 0.]), p)
        positions.append(np.array(s.agent_pos))
        if i == 0:
            assert not done and float(s.speed) > 0
        if bool(done):
            break
    assert done and info['goal_reached'] and info['terminated'] and not info['truncated']
    assert float(s.speed) <= p.goal_speed
    assert np.linalg.norm(positions[0] - positions[-1]) > 0
    assert float(jnp.linalg.norm(s.agent_pos - s.goal_pos)) <= p.goal_radius


def test_velocity_sensing_distinguishes_an_approaching_obstacle():
    p, s = RobotaxiEnv.init_params(KEY, map_id=3)
    s = change(s, agent_pos=jnp.array([10., 10.]), heading=0., agent_forward_dir=jnp.array([1., 0.]),
               speed=1., static_obstacles=jnp.empty((0, 2)),
               kinematic_obstacles=jnp.array([[15., 10.]]),
               kinematic_obst_velocities=jnp.array([[-.5, 0.]]))
    obs = RobotaxiEnv.get_observation(s, p)
    front = p.num_ray_sensors // 2
    assert obs.as_vector().shape == (70,)
    assert float(obs.ray_distances[front]) == pytest.approx(4 / p.perception_radius)
    assert float(obs.ray_closing_speeds[front]) == pytest.approx(1.5 / 5.5)
    s = change(s, kinematic_obst_velocities=jnp.array([[2., 0.]]))
    receding = RobotaxiEnv.get_observation(s, p)
    assert float(receding.ray_closing_speeds[front]) < 0


def test_empty_scene_has_fixed_finite_observation():
    p, s = RobotaxiEnv.init_params(KEY, map_id=3)
    s = change(s, static_obstacles=jnp.empty((0, 2)), kinematic_obstacles=jnp.empty((0, 2)),
               kinematic_obst_velocities=jnp.empty((0, 2)))
    obs = RobotaxiEnv.get_observation(s, p)
    assert obs.as_vector().shape == (RobotaxiEnv.observation_size(p),)
    assert np.isfinite(obs.as_vector()).all()
    np.testing.assert_array_equal(obs.ray_closing_speeds, np.zeros(32))


def test_mixed_batched_autoresets_preserve_terminal_observation_and_bootstrap():
    p, initial = RobotaxiEnv.init_params(KEY, map_id=3, max_steps=10)
    goal = change(initial, agent_pos=initial.goal_pos)
    collision = change(initial, agent_pos=initial.static_obstacles[0])
    timeout = change(initial, time=jnp.array(9, dtype=jnp.int32))
    states = jax.tree.map(lambda *x: jnp.stack(x), goal, collision, timeout, initial)
    keys = jax.random.split(KEY, 4)
    actions = jnp.zeros((4, 2))
    env = AutoResetWrapper(RobotaxiEnv, p, initial)
    batch = jax.jit(jax.vmap(env.step))(keys, states, actions)
    obs, new, reward, done, info = batch
    np.testing.assert_array_equal(done, [True, True, True, False])
    np.testing.assert_array_equal(info['terminated'], [True, True, False, False])
    np.testing.assert_array_equal(info['truncated'], [False, False, True, False])
    np.testing.assert_array_equal(new.time, [0, 0, 0, 1])
    assert info['final_observation'].shape == (4, 70)
    raw = jax.jit(jax.vmap(lambda key, state, action: RobotaxiEnv.step(key, state, action, p)))(keys, states, actions)
    np.testing.assert_allclose(info['final_observation'], jax.vmap(lambda x: x.as_vector())(raw[0]))
    assert obs.as_vector().shape == (4, 70)
    np.testing.assert_allclose(obs.speed[:3], np.zeros(3))
    np.testing.assert_allclose(1 - np.asarray(info['terminated']), [0, 0, 1, 1])
    assert float(reward[0]) > 99 and float(reward[1]) < -99


def test_randomized_resets_are_reproducible_and_change_with_key():
    p, initial = RobotaxiEnv.init_params(KEY, map_id=0)
    one = RobotaxiEnv.reset(KEY, p, initial, randomize=True)
    again = RobotaxiEnv.reset(KEY, p, initial, randomize=True)
    for a, b in zip(jax.tree.leaves(one), jax.tree.leaves(again)):
        np.testing.assert_array_equal(a, b)
    other = RobotaxiEnv.reset(jax.random.PRNGKey(8), p, initial, randomize=True)
    assert not np.array_equal(one[0].as_vector(), other[0].as_vector())
    obs, states = jax.jit(jax.vmap(lambda key: RobotaxiEnv.reset(key, p, initial, randomize=True)))(jax.random.split(KEY, 32))
    assert obs.ray_distances.shape == (32, 32)
    assert np.isfinite(jax.vmap(lambda x: x.as_vector())(obs)).all()
    assert float(jnp.max(states.speed)) <= .5 and int(jnp.max(states.time)) == 0


def test_timeout_flags_remain_consistent_in_absorbing_state():
    p, s = RobotaxiEnv.init_params(KEY, map_id=3, max_steps=1)
    _, s, _, done, info = RobotaxiEnv.step(KEY, s, jnp.zeros(2), p)
    assert done and info['truncated'] and not info['terminated']
    _, _, reward, done, info = RobotaxiEnv.step(KEY, s, jnp.zeros(2), p)
    assert done and info['truncated'] and not info['terminated'] and float(reward) == 0


def test_invalid_action_shape_has_clear_error():
    p, s = RobotaxiEnv.init_params(KEY)
    with pytest.raises(ValueError, match='shape'):
        RobotaxiEnv.step(KEY, s, jnp.ones(3), p)
