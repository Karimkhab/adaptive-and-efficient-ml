import math
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env.base import BaseEnv
from env.robotaxi import RobotaxiEnv, vehicle_motion, swept_clearance, plan_path
from utils.autoreset import AutoResetWrapper

KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def env():
    return RobotaxiEnv.init_params(KEY, map_id=3, max_steps=100)


def replace(state, **values):
    for name, value in values.items():
        state = eqx.tree_at(lambda x: getattr(x, name), state, jnp.asarray(value))
    return state


def test_interface_reset_and_finite_observation(env):
    p, s = env
    assert issubclass(RobotaxiEnv, BaseEnv)
    obs, reset = RobotaxiEnv.reset(KEY, p, s)
    assert obs.as_vector().shape == (38,)
    assert np.isfinite(obs.as_vector()).all()
    assert obs.collision_rays.shape == (32, 2)
    np.testing.assert_array_equal(reset.agent_pos, s.agent_pos)


def test_standstill_steering_and_braking_do_not_move(env):
    p, s = env
    for action in ([0., 1.], [-1., -1.]):
        pos, heading, speed = vehicle_motion(s.agent_pos, s.heading, s.speed, jnp.array(action), p)
        np.testing.assert_allclose(pos, s.agent_pos)
        assert float(speed) == 0.
        assert float(heading) == float(s.heading)


def test_acceleration_matches_first_step(env):
    p, s = env
    pos, heading, speed = vehicle_motion(s.agent_pos, s.heading, s.speed, jnp.array([1., 0.]), p)
    assert float(speed) == pytest.approx(p.max_accel * p.step_size)
    assert float(pos[0] - s.agent_pos[0]) == pytest.approx(.5 * p.max_accel * p.step_size**2, abs=1e-6)
    assert float(heading) == 0.


@pytest.mark.parametrize("steering,sign", [(1., 1.), (-1., -1.)])
def test_turn_direction_and_bicycle_rate(env, steering, sign):
    p, s = env
    _, heading, speed = vehicle_motion(s.agent_pos, s.heading, jnp.array(2.), jnp.array([0., steering]), p)
    expected = (2 + float(speed)) / 2 * math.tan(steering * p.max_steer) / p.wheelbase * p.step_size
    assert float(heading) * sign > 0
    assert float(heading) == pytest.approx(expected, abs=1e-6)


def test_drag_and_brakes_reduce_speed_without_reverse(env):
    p, s = env
    _, _, coasting = vehicle_motion(s.agent_pos, s.heading, jnp.array(2.), jnp.array([0., 0.]), p)
    _, _, braking = vehicle_motion(s.agent_pos, s.heading, jnp.array(2.), jnp.array([-1., 0.]), p)
    assert 0 < float(braking) < float(coasting) < 2
    _, _, stopped = vehicle_motion(s.agent_pos, s.heading, jnp.array(.01), jnp.array([-1., 0.]), p)
    assert float(stopped) == 0


def test_action_and_speed_limits(env):
    p, s = env
    a = vehicle_motion(s.agent_pos, s.heading, jnp.array(p.max_speed), jnp.array([10., 8.]), p)
    b = vehicle_motion(s.agent_pos, s.heading, jnp.array(p.max_speed), jnp.array([1., 1.]), p)
    for left, right in zip(a, b):
        np.testing.assert_allclose(left, right)
    assert float(a[2]) == p.max_speed


def test_goal_reward_once_and_absorbing_state(env):
    p, s = env
    s = replace(s, agent_pos=s.goal_pos)
    obs, terminal, reward, done, info = RobotaxiEnv.step(KEY, s, jnp.zeros(2), p)
    assert done and info["goal_reached"] and not info["collision"]
    assert float(reward) == pytest.approx(p.goal_bonus - p.time_cost * p.step_size, abs=1e-5)
    _, again, second, done, info = RobotaxiEnv.step(KEY, terminal, jnp.ones(2), p)
    assert done and float(second) == 0 and not info["goal_reached"]
    for old, new in zip(jax.tree.leaves(terminal), jax.tree.leaves(again)):
        np.testing.assert_array_equal(old, new)
    assert np.isfinite(obs.as_vector()).all()


def test_collision_has_priority_over_goal(env):
    p, s = env
    s = replace(s, agent_pos=s.static_obstacles[0], goal_pos=s.static_obstacles[0])
    _, _, reward, done, info = RobotaxiEnv.step(KEY, s, jnp.zeros(2), p)
    assert done and info["collision"] and not info["goal_reached"]
    assert float(reward) < -99


def test_exact_timeout_and_autoreset(env):
    p, s = env
    s = replace(s, time=jnp.array(p.max_steps_in_episode - 1, dtype=jnp.int32))
    _, end, _, done, info = RobotaxiEnv.step(KEY, s, jnp.zeros(2), p)
    assert done and info["truncated"] and int(end.time) == p.max_steps_in_episode
    wrapper = AutoResetWrapper(RobotaxiEnv, p, env[1])
    obs, reset, reward, done, info = wrapper.step(KEY, s, jnp.zeros(2))
    assert done and info["truncated"] and int(reset.time) == 0
    assert float(reward) < 0
    np.testing.assert_array_equal(reset.agent_pos, env[1].agent_pos)
    assert np.isfinite(obs.as_vector()).all()


def test_swept_collision_catches_moving_object_crossing():
    agent = jnp.array([5., 5.])
    before, after = jnp.array([[1., 5.]]), jnp.array([[9., 5.]])
    assert float(swept_clearance(agent, agent, before, after, 1.)) == pytest.approx(-2.)
    assert float(jnp.linalg.norm(before[0] - agent)) > 2
    assert float(jnp.linalg.norm(after[0] - agent)) > 2


def test_contact_geometry_is_sum_of_radii():
    agent = jnp.zeros(2)
    obstacle = jnp.array([[2., 0.]])
    assert float(swept_clearance(agent, agent, obstacle, obstacle, 1.)) == pytest.approx(0.)


def test_progress_reward_uses_resulting_transition(env):
    p, s = env
    _, new, reward, done, _ = RobotaxiEnv.step(KEY, s, jnp.array([1., 0.]), p)
    progress = np.linalg.norm(s.agent_pos - s.goal_pos) - np.linalg.norm(new.agent_pos - s.goal_pos)
    assert not done
    assert float(reward) == pytest.approx(progress - p.time_cost * p.step_size, abs=1e-6)


@pytest.mark.parametrize("map_id", [0, 1, 2, 3])
def test_maps_have_valid_clear_start_and_path(map_id):
    p, s = RobotaxiEnv.init_params(KEY, map_id=map_id)
    obstacles = jnp.concatenate([s.static_obstacles, s.kinematic_obstacles])
    assert float(jnp.min(jnp.linalg.norm(s.agent_pos - obstacles, axis=1))) > 2 * p.radius
    np.testing.assert_allclose(s.path_array[-1], s.goal_pos)
    assert np.isfinite(s.path_array).all()
    for point in np.asarray(s.path_array):
        assert float(jnp.min(jnp.linalg.norm(point - s.static_obstacles, axis=1))) > 2 * p.radius


def test_jit_eager_and_vmap_agree(env):
    p, s = env
    action = jnp.array([.5, .3])
    compiled = RobotaxiEnv.step(KEY, s, action, p)
    with jax.disable_jit():
        eager = RobotaxiEnv.step(KEY, s, action, p)
    for a, b in zip(jax.tree.leaves(compiled), jax.tree.leaves(eager)):
        np.testing.assert_allclose(a, b, atol=2e-5)
    states = jax.tree.map(lambda x: jnp.stack([x, x]), s)
    batched = jax.jit(jax.vmap(lambda st, ac: RobotaxiEnv.step(KEY, st, ac, p)))(states, jnp.stack([action, action]))
    np.testing.assert_allclose(batched[1].agent_pos[0], compiled[1].agent_pos)


def test_repeated_goal_waypoints_are_finite(env):
    p, s = env
    s = replace(s, path_array=jnp.tile(s.goal_pos, (100, 1)), agent_pos=s.goal_pos)
    obs = RobotaxiEnv.get_observation(s, p)
    assert np.isfinite(obs.as_vector()).all()


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        RobotaxiEnv.init_params(KEY, cell_size=2.)
    with pytest.raises(ValueError):
        RobotaxiEnv.init_params(KEY, max_steps=0)


@pytest.mark.parametrize("map_id", [0, 1, 2, 3])
def test_complete_demo_rollout_is_finite_and_collision_free(map_id):
    from evaluate import rollout
    result, trajectory, _ = rollout(map_id)
    assert result["goal_reached"]
    assert not result["collision"] and not result["truncated"]
    assert result["min_clearance"] > 0
    assert np.isfinite(trajectory).all()
