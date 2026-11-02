import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ppo import (advantages, init_network, sample, forward, log_prob, ppo_loss,
                 make_update, save_checkpoint, load_checkpoint)
from rl_env import make_env, route_potential
from env.robotaxi import RobotaxiEnv


def test_gae_distinguishes_timeout_terminal_and_reset():
    reward = jnp.array([[1., 1.], [2., 2.]])
    value = jnp.zeros_like(reward)
    next_value = jnp.full_like(reward, 10.)
    terminal = jnp.array([[False, True], [True, True]])
    done = jnp.ones_like(terminal)
    adv, target = advantages(reward, value, next_value, terminal, done, gamma=.9)
    # First column is a timeout; second is a true terminal. Neither may
    # borrow the next episode's return (2).
    np.testing.assert_allclose(adv, [[10., 1.], [2., 2.]])


def test_gae_finite_horizon_by_hand():
    reward = jnp.array([[1.], [2.], [3.]])
    adv, _ = advantages(reward, jnp.zeros_like(reward), jnp.zeros_like(reward),
                        jnp.zeros_like(reward), jnp.zeros_like(reward), gamma=.5, lam=1.)
    np.testing.assert_allclose(adv[:, 0], [2.75, 3.5, 3.])


def test_action_repeat_matches_raw_physics_and_preserves_timeout():
    p, initial, reset, step = make_env(3, repeat=6, max_steps=4)
    action = jnp.array([.6, .1])
    state, reward = initial, 0.
    for _ in range(4):
        _, state, r, _, _ = RobotaxiEnv.step(jax.random.PRNGKey(0), state, action, p)
        reward += float(r)
    end, final_obs, macro_reward, done, info = jax.jit(step)(initial, action)
    assert done and info["timeout"] and not info["success"]
    assert int(info["physics_steps"]) == 4
    assert int(end.time) == 4
    np.testing.assert_allclose(end.agent_pos, state.agent_pos, atol=1e-6)
    np.testing.assert_allclose(macro_reward, reward, atol=1e-6)
    np.testing.assert_allclose(final_obs, RobotaxiEnv.get_observation(end, p).as_vector(), atol=1e-6)


def test_sampling_density_ratio_and_finite_gradients():
    params = init_network(0)
    obs = jax.random.normal(jax.random.PRNGKey(1), (16, 70))
    action, latent, lp, values = sample(params, obs, jax.random.PRNGKey(2))
    assert np.isfinite(action).all() and np.all(np.abs(action) < 1)
    mean, _ = forward(params, obs)
    np.testing.assert_allclose(jnp.exp(log_prob(params, mean, latent) - lp), 1.)
    batch = (obs, latent, lp, values, jnp.linspace(-1., 1., 16), values + .3)
    (loss, stats), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, batch)
    assert np.isfinite(loss)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    assert sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(grads)) > 0
    optimizer, update = make_update(epochs=1, minibatches=2)
    updated, _, metrics = update(params, optimizer.init(params), batch, jax.random.PRNGKey(3))
    assert not np.array_equal(updated["actor"][-1]["w"], params["actor"][-1]["w"])


def test_checkpoint_exact_roundtrip(tmp_path):
    params = init_network(42)
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, params, {"seed": 42, "hidden": 64, "observation_version": 2, "obs_dim": 70})
    restored, _ = load_checkpoint(path)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(a, b)


def test_route_progress_does_not_confuse_required_detour_with_regression():
    # Goal is east, but a wall requires going north first.
    path = jnp.array([[0., 0.], [0., 3.], [3., 3.], [3., 0.]])
    start, north = jnp.array([0., 0.]), jnp.array([0., 1.])
    assert float(jnp.linalg.norm(north - path[-1])) > float(jnp.linalg.norm(start - path[-1]))
    assert float(route_potential(north, path) - route_potential(start, path)) == pytest.approx(1.)
    assert float(route_potential(path[-1], path)) == pytest.approx(0.)


def test_training_reward_change_keeps_physics_and_raw_score_identical():
    _, initial, _, base = make_env(0)
    _, _, _, shaped = make_env(0, route_reward=True)
    action = jnp.array([.8, 0.])
    a, obs_a, reward_a, done_a, info_a = jax.jit(base)(initial, action)
    b, obs_b, reward_b, done_b, info_b = jax.jit(shaped)(initial, action)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert float(reward_a) == float(info_b["raw_reward"])
    assert float(reward_b) > float(reward_a)


def test_checkpoint_rejects_corruption_and_legacy_contract(tmp_path):
    import json
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, init_network(0), {"seed": 0, "hidden": 64, "observation_version": 2, "obs_dim": 70})
    blob = bytearray(path.read_bytes())
    blob[-5] ^= 1
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path)
    path.with_suffix(".json").write_text(json.dumps({"seed": 0, "hidden": 64, "observation_version": 1, "obs_dim": 38}))
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(path)


def test_inference_uses_only_a_single_observation(tmp_path):
    from policy import Policy
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, init_network(0), {"seed": 0, "hidden": 64, "observation_version": 2, "obs_dim": 70})
    actor = Policy(path)
    action = actor(jnp.zeros(70))
    assert action.shape == (2,) and np.isfinite(action).all()
    assert np.all(np.abs(action) <= 1)
    with pytest.raises(ValueError, match="70"):
        actor(jnp.zeros(38))


def test_safety_objective_does_not_modify_actions_physics_or_raw_score():
    _, initial, _, base=make_env(2)
    _, _, _, shaped=make_env(2,route_reward=True,parking_reward=True,cruise_speed=2.5,
        speed_cost=1.,clearance_cost=4.,clearance_margin=1.,clearance_speed=True)
    action=jnp.array([.7,.4])
    original=jax.jit(base)(initial,action)
    safe=jax.jit(shaped)(initial,action)
    for left,right in zip(jax.tree.leaves(original[0]),jax.tree.leaves(safe[0])):
        np.testing.assert_array_equal(left,right)
    np.testing.assert_array_equal(original[1],safe[1])
    np.testing.assert_array_equal(original[2],safe[4]['raw_reward'])
    assert float(original[2])!=float(safe[2])
