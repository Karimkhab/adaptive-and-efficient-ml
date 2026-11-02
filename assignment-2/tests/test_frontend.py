"""Exercise real input mapping and the same offscreen renderer used by video."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from collections import defaultdict
import jax
import numpy as np
import pygame
import pytest

from env.robotaxi import RobotaxiEnv
from utils.taxi_renderer import TaxiFrontend


@pytest.fixture
def frontend():
    p, state = RobotaxiEnv.init_params(jax.random.PRNGKey(0), map_id=3)
    ui = TaxiFrontend(p, state, headless=True)
    yield ui
    pygame.quit()


def key(code):
    return pygame.event.Event(pygame.KEYDOWN, key=code)


def test_toggle_pause_map_reset_and_manual_takeover(frontend):
    ui = frontend
    assert not ui.demo
    ui.handle_event(key(pygame.K_TAB))
    assert ui.demo
    ui.handle_event(key(pygame.K_SPACE))
    assert ui.paused
    ui.handle_event(key(pygame.K_l))
    assert ui.show_rays
    ui.handle_event(key(pygame.K_c))
    assert not ui.show_colliders
    ui.handle_event(key(pygame.K_z))
    assert not ui.follow_camera
    ui.handle_event(key(pygame.K_4))
    assert ui.map_id == 3 and int(ui.state.time) == 0 and not ui.paused
    ui.info = {"truncated": True}
    import equinox as eqx
    import jax.numpy as jnp
    ui.state = eqx.tree_at(lambda s: s.terminated, ui.state, jnp.array(True))
    assert ui.handle_event(key(pygame.K_TAB))
    assert not ui.demo and not bool(ui.state.terminated)


def test_keyboard_drives_the_real_vehicle(frontend, monkeypatch):
    ui = frontend
    pressed = defaultdict(bool, {pygame.K_w: True, pygame.K_a: True})
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: pressed)
    action = ui.handle_keys()
    np.testing.assert_array_equal(action, [1., 1.])
    original = np.array(ui.state.agent_pos)
    for _ in range(15):
        ui.obs, ui.state, _, done, ui.info = RobotaxiEnv.step(ui.key, ui.state, action, ui.params)
    assert float(ui.state.speed) > 0
    assert float(ui.state.heading) > 0
    assert not np.array_equal(ui.state.agent_pos, original)
    assert not done


def test_render_is_a_real_frame_and_pause_does_not_extend_trail(frontend):
    ui = frontend
    before = np.array(ui.state.agent_pos)
    ui.draw(3, "Manual drive")
    frame = ui.frame()
    assert frame.shape == (800, 1280, 3) and frame.dtype == np.uint8
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 20
    trail_size = len(ui.history)
    ui.draw(3, "Manual drive")
    assert len(ui.history) == trail_size
    np.testing.assert_array_equal(ui.state.agent_pos, before)
