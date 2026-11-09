"""Campus taxi frontend. Decorative car bodies fit inside the circle colliders."""
from collections import deque
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from utils.renderer import PygameFrontend
from controller import demo_action
from env.robotaxi import RobotaxiEnv


SCENES = ["Campus detour", "Inner courtyard", "Service lane", "Crossing traffic", "Island detour"]
INK = (25, 37, 49)
MUTED = (103, 119, 130)
AMBER = (250, 193, 66)
TEAL = (91, 204, 187)


class TaxiFrontend(PygameFrontend):
    def __init__(self, params, initial, headless=False):
        pygame.init()
        self.params, self.init_state = params, initial
        self.headless = headless
        self.width, self.height = 1280, 800
        self.screen = pygame.Surface((self.width, self.height)) if headless else pygame.display.set_mode((self.width, self.height))
        if not headless:
            pygame.display.set_caption("Campus Taxi — PIXEL / JAX Environment")
        self.font = pygame.font.Font(None, 25)
        self.small = pygame.font.Font(None, 20)
        self.title = pygame.font.Font(None, 38)
        self.big = pygame.font.Font(None, 52)
        self.world = pygame.Rect(24, 108, 960, 532)
        self.sidebar = pygame.Rect(1008, 108, 248, 532)
        self.clock = pygame.time.Clock()
        self.key = jax.random.PRNGKey(0)
        self.obs, self.state = RobotaxiEnv.reset(self.key, params, initial)
        self.info = {}
        self.running, self.demo, self.paused, self.show_rays = True, False, False, False
        self.show_colliders = True
        self.follow_camera = True
        self.action = np.zeros(2)
        self.history = deque(maxlen=1800)
        self.last_time = -1
        self.map_id = 0
        self.playback_speed = 1
        self._frame_scale = None
        self._frame_center = None

    @property
    def scale(self):
        if self._frame_scale is not None:
            return self._frame_scale
        h, w = self.params.map_height_width
        overview = min((self.world.w - 48) / ((w - 1) * self.params.cell_size + 2 * self.params.radius),
                       (self.world.h - 48) / ((h - 1) * self.params.cell_size + 2 * self.params.radius))
        return overview * (2 if self.follow_camera else 1)

    @property
    def camera_center(self):
        if self._frame_center is not None:
            return self._frame_center
        extent = (np.array(self.params.map_height_width[::-1]) - 1) * self.params.cell_size
        if not self.follow_camera:
            return extent / 2
        half_view = np.array(self.world.size) / (2 * self.scale)
        low, high = half_view - self.params.radius, extent - half_view + self.params.radius
        return np.where(high >= low, np.clip(np.asarray(self.state.agent_pos), low, high), extent / 2)

    def point(self, xy):
        x, y = np.asarray(xy)
        cx, cy = self.camera_center
        return (round(self.world.centerx + (x - cx) * self.scale),
                round(self.world.centery - (y - cy) * self.scale))

    def minimap(self):
        box = pygame.Rect(790, 14, 236, 80)
        pygame.draw.rect(self.screen, (29, 41, 52), box, border_radius=8)
        extent = (np.array(self.params.map_height_width[::-1]) - 1) * self.params.cell_size
        scale = min((box.w - 12) / (extent[0] + 2), (box.h - 12) / (extent[1] + 2))
        def point(pos):
            delta = np.asarray(pos) - extent / 2
            return (round(box.centerx + delta[0] * scale), round(box.centery - delta[1] * scale))
        for pos in np.asarray(self.state.static_obstacles):
            pygame.draw.circle(self.screen, (102, 124, 108), point(pos), 1)
        path = [point(pos) for pos in np.asarray(self.state.path_array)]
        pygame.draw.lines(self.screen, (65, 111, 117), False, path, 1)
        pygame.draw.circle(self.screen, TEAL, point(self.state.goal_pos), 3, 1)
        pygame.draw.circle(self.screen, AMBER, point(self.state.agent_pos), 3)
        cx, cy = self.camera_center
        half_width, half_height = np.array(self.world.size) / (2 * self.scale)
        a, b = point([cx - half_width, cy + half_height]), point([cx + half_width, cy - half_height])
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(box)
        pygame.draw.rect(self.screen, (166, 180, 179), pygame.Rect(a, (b[0] - a[0], b[1] - a[1])), 1)
        self.screen.set_clip(previous_clip)

    def label(self, text, position, color=INK, small=False, font=None):
        self.screen.blit((font or (self.small if small else self.font)).render(str(text), True, color), position)

    def car(self, position, heading, color, steering=0., braking=False):
        """8x8 code-drawn sprite, nearest-neighbour scaled inside radius r."""
        sprite = pygame.Surface((8, 8), pygame.SRCALPHA)
        dark = (19, 30, 37)
        pygame.draw.rect(sprite, dark, (1, 2, 6, 4))
        pygame.draw.rect(sprite, color, (2, 2, 4, 4))
        pygame.draw.rect(sprite, color, (6, 3, 1, 2))
        pygame.draw.rect(sprite, (255, 231, 165), (6, 2, 1, 1))
        pygame.draw.rect(sprite, (255, 231, 165), (6, 5, 1, 1))
        pygame.draw.rect(sprite, (65, 105, 120), (4, 2, 1, 4))
        pygame.draw.rect(sprite, (32, 64, 79), (2, 3, 1, 2))
        pygame.draw.rect(sprite, (255, 83, 66) if braking else (143, 63, 53), (1, 3, 1, 2))
        for x in (2, 5):
            for y in (1, 6):
                wheel = steering if x == 5 else 0.
                dx, dy = math.cos(wheel), -math.sin(wheel)
                pygame.draw.line(sprite, dark, (round(x - .6 * dx), round(y - .6 * dy)),
                                 (round(x + .6 * dx), round(y + .6 * dy)), 1)
        # Nearest-neighbour rotation preserves hard pixels; all artwork fits
        # inside the circular physical collider. Pixel positions never feed physics.
        pixel_scale = max(1, int(self.params.radius * self.scale / 4))
        rotated = pygame.transform.rotate(sprite, math.degrees(heading))
        image = pygame.transform.scale(rotated, (rotated.get_width() * pixel_scale,
                                                  rotated.get_height() * pixel_scale))
        self.screen.blit(image, image.get_rect(center=self.point(position)))
        if self.show_colliders:
            pygame.draw.circle(self.screen, tuple(min(255, c + 12) for c in color),
                               self.point(position), round(self.params.radius * self.scale), 1)

    def draw(self, map_id=0, mode="Manual drive", episode_return=0.):
        s, p = self.state, self.params
        self._frame_scale, self._frame_center = None, None
        scale = self.scale
        center = self.camera_center
        self._frame_scale, self._frame_center = scale, center
        is_demo = self.demo or "heuristic" in mode.lower()
        if "action" in self.info:
            self.action = np.asarray(self.info["action"])
        current_time = int(s.time)
        if current_time < self.last_time:
            self.history.clear()
        if current_time != self.last_time:
            self.history.append(np.asarray(s.agent_pos).copy())
            self.last_time = current_time
        self.screen.fill((233, 238, 240))
        self.label("CAMPUS TAXI", (28, 24), font=self.title)
        self.label("JAX / VEHICLE ENVIRONMENT", (30, 66), MUTED, small=True)
        scene = SCENES[map_id]
        if map_id == 4 and self.state.kinematic_obstacles.shape[0]:
            scene += " + traffic"
        self.label(f"{map_id + 1:02d}  /  {scene}", (410, 37))
        if self.playback_speed != 1:
            self.label(f"{self.playback_speed}x PLAYBACK", (410, 70), MUTED, small=True)
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = TEAL
        if bool(s.done):
            status = "PARKED" if bool(self.info.get("goal_reached", False)) else (
                "COLLISION" if bool(self.info.get("collision", False)) else "TIME LIMIT")
            status_color = TEAL if status == "PARKED" else (245, 124, 102)
        pygame.draw.rect(self.screen, (26, 43, 54), (1052, 24, 204, 54), border_radius=14)
        self.label(status, (1072, 42), status_color)
        pygame.draw.rect(self.screen, (29, 41, 52), self.world, border_radius=20)
        pygame.draw.rect(self.screen, (249, 251, 250), self.sidebar, border_radius=20)
        self.screen.set_clip(self.world)

        # Faint survey grid; obstacle artwork remains inside the true circles.
        h, w = p.map_height_width
        for y in range(self.world.top + 8, self.world.bottom - 8, 16):
            for x in range(self.world.left + 8, self.world.right - 8, 16):
                if (x * 7 + y * 11) % 13 < 3:
                    pygame.draw.rect(self.screen, (35, 47, 57), (x, y, 2, 2))
        radius = max(2, round(p.radius * self.scale))
        planter = pygame.Surface((8, 8), pygame.SRCALPHA)
        pygame.draw.rect(planter, (53, 66, 65), (1, 1, 6, 6))
        pygame.draw.rect(planter, (127, 139, 119), (2, 1, 4, 1))
        pygame.draw.rect(planter, (88, 112, 84), (1, 2, 6, 4))
        pygame.draw.rect(planter, (110, 135, 95), (2, 2, 3, 2))
        pygame.draw.rect(planter, (70, 93, 74), (4, 4, 2, 2))
        pixel_scale = max(1, int(p.radius * self.scale / 4))
        planter = pygame.transform.scale(planter, (8 * pixel_scale, 8 * pixel_scale))
        for position in np.asarray(s.static_obstacles):
            center = self.point(position)
            self.screen.blit(planter, planter.get_rect(center=center))
            if self.show_colliders:
                pygame.draw.circle(self.screen, (96, 116, 103), center, radius, 1)
        path = [self.point(v) for v in np.asarray(s.path_array)]
        for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
            if a != b:
                pygame.draw.line(self.screen, (65, 111, 117), a, b, 2)
                pygame.draw.circle(self.screen, TEAL, b, 2)
        if len(self.history) > 1:
            pygame.draw.lines(self.screen, (185, 139, 59), False, [self.point(pos) for pos in self.history], 3)
        if self.show_rays:
            layer = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            layer.set_clip(self.world.inflate(-2, -2))
            for ray in np.asarray(self.obs.collision_rays):
                pygame.draw.line(layer, (118, 211, 201, 40), self.point(s.agent_pos), self.point(np.asarray(s.agent_pos) + ray), 1)
            self.screen.blit(layer, (0, 0))
        for position, velocity in zip(np.asarray(s.kinematic_obstacles), np.asarray(s.kinematic_obst_velocities)):
            self.car(position, math.atan2(velocity[1], velocity[0]), (225, 126, 107))
        goal = self.point(s.goal_pos)
        pygame.draw.circle(self.screen, TEAL, goal, round(p.goal_radius * self.scale), 2)
        pygame.draw.circle(self.screen, (50, 120, 113), goal, round(1.4 * p.goal_radius * self.scale), 1)
        pygame.draw.rect(self.screen, (30, 75, 76), (goal[0] - 6, goal[1] - 6, 12, 12))
        for dx, dy in ((-2, -4), (0, -4), (-2, -2), (2, -2), (-2, 0), (0, 0), (-2, 2), (-2, 4)):
            pygame.draw.rect(self.screen, (126, 225, 190), (goal[0] + dx, goal[1] + dy, 2, 2))
        self.label("DESTINATION", (goal[0] - 39, goal[1] - 37), TEAL, small=True)
        self.car(s.agent_pos, float(s.heading), AMBER,
                 float(self.action[1]) * p.max_steer, bool(self.action[0] < -.1))

        # One 480x266 logical game canvas, exactly 2x nearest-neighbour.
        # HUD stays at native resolution for readable text.
        game = self.screen.subsurface(self.world).copy()
        game = pygame.transform.scale(game, (self.world.w // 2, self.world.h // 2))
        self.screen.blit(pygame.transform.scale(game, self.world.size), self.world)
        self.screen.set_clip(None)
        self.minimap()

        x = self.sidebar.x + 20
        is_ppo = "ppo" in mode.lower()
        self.label("RL / PPO" if is_ppo else ("DEMO / HEURISTIC" if is_demo else "MANUAL / KEYBOARD"), (x, 130), MUTED, small=True)
        self.label("MODEL SPEED", (x, 172), MUTED, small=True)
        self.label(f"{float(s.speed):.2f}", (x, 197), font=self.big)
        self.label("units / second", (x, 245), MUTED, small=True)
        pygame.draw.rect(self.screen, (221, 230, 228), (x, 277, 208, 7), border_radius=3)
        pygame.draw.rect(self.screen, (58, 150, 134), (x, 277, round(208 * float(s.speed) / p.max_speed), 7), border_radius=3)
        rows = [
            ("TIME", f"{current_time * p.step_size:.1f} s"),
            ("GOAL DISTANCE", f"{float(np.linalg.norm(s.goal_pos - s.agent_pos)):.2f} u"),
            ("CLEARANCE", f"{float(self.info['clearance']):.2f} u" if "clearance" in self.info and math.isfinite(float(self.info['clearance'])) else "--"),
            ("RETURN", f"{episode_return:.2f}"),
        ]
        for i, (name, value) in enumerate(rows):
            y = 311 + i * 48
            self.label(name, (x, y), MUTED, small=True)
            self.label(value, (x, y + 18))
        pygame.draw.line(self.screen, (220, 228, 228), (x, 512), (x + 208, 512))
        self.label(f"GAS / BRAKE    {float(self.action[0]):+.2f}", (x, 532), small=True)
        self.label(f"STEERING       {math.degrees(float(self.action[1]) * p.max_steer):+.1f} deg", (x, 558), small=True)
        self.label("Trained neural policy" if is_ppo else ("No trained policy" if is_demo else "You control the vehicle"), (x, 605), MUTED, small=True)

        pygame.draw.rect(self.screen, (249, 251, 250), (24, 660, 1232, 116), border_radius=18)
        self.label("PPO / observed features only   Z  Zoom   L  Rays   C  Colliders   ESC  Exit" if is_ppo else "TAB  Demo/manual   SPACE  Pause   R  Reset   1-5  Map   Z  Zoom   L  Rays   C  Colliders   ESC  Exit", (46, 683))
        self.label(mode if is_ppo else "W / Up: accelerate    S / Down: brake    A / Left, D / Right: steer", (46, 716), MUTED, small=True)
        self.label("Gold: taxi + trail    Coral: moving traffic    Grey: static obstacles    Teal: reference path / goal", (46, 743), MUTED, small=True)
        if not self.headless:
            pygame.display.flip()

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        throttle = int(keys[pygame.K_UP] or keys[pygame.K_w]) - int(keys[pygame.K_DOWN] or keys[pygame.K_s])
        steering = int(keys[pygame.K_LEFT] or keys[pygame.K_a]) - int(keys[pygame.K_RIGHT] or keys[pygame.K_d])
        return jnp.array([throttle, steering], dtype=jnp.float32)

    def reset_episode(self, map_id=None):
        if map_id is not None:
            self.map_id = map_id
            self.params, self.init_state = RobotaxiEnv.init_params(
                self.key, map_id=map_id, max_steps=self.params.max_steps_in_episode,
                fps=self.params.fps,
            )
        self.obs, self.state = RobotaxiEnv.reset(self.key, self.params, self.init_state)
        self.info, self.action = {}, np.zeros(2)
        self.paused = False
        self.history.clear()
        self.last_time = -1

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        if event.type != pygame.KEYDOWN:
            return False
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_TAB:
            self.demo = not self.demo
            if bool(self.state.done):
                self.reset_episode()
                return True
        elif event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif event.key == pygame.K_c:
            self.show_colliders = not self.show_colliders
        elif event.key == pygame.K_z:
            self.follow_camera = not self.follow_camera
        elif event.key == pygame.K_l:
            self.show_rays = not self.show_rays
        elif event.key == pygame.K_r:
            self.reset_episode()
            return True
        elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
            self.reset_episode(event.key - pygame.K_1)
            return True
        return False

    def frame(self):
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def run(self, map_id=0, demo=False, record=None):
        os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
        import imageio.v2 as imageio
        self.map_id, self.demo = map_id, demo
        if record:
            Path(record).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(record), fps=self.params.fps, codec="libx264", quality=7) if record else None
        episode_return = 0.
        try:
            while self.running:
                for event in pygame.event.get():
                    if self.handle_event(event):
                        episode_return = 0.
                if not self.running:
                    break
                if not self.paused and not bool(self.state.done):
                    action = demo_action(self.state, self.params) if self.demo else self.handle_keys()
                    self.obs, self.state, reward, _, self.info = RobotaxiEnv.step(self.key, self.state, action, self.params)
                    self.info["action"] = action
                    episode_return += float(reward)
                self.draw(self.map_id, "Heuristic controller" if self.demo else "Manual drive", episode_return)
                if writer:
                    writer.append_data(self.frame())
                if bool(self.state.done) and (record or self.headless):
                    if writer:
                        for _ in range(self.params.fps * 2):
                            writer.append_data(self.frame())
                    break
                if not self.headless:
                    self.clock.tick(self.params.fps)
        finally:
            if writer:
                writer.close()
            pygame.quit()
