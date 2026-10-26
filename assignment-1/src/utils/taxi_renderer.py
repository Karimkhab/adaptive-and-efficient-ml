"""Frontend adapted from the supplied PygameFrontend for world-unit circles."""
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


class TaxiFrontend(PygameFrontend):
    def __init__(self, params, initial, headless=False):
        pygame.init()
        self.params, self.init_state = params, initial
        self.headless = headless
        self.width, self.height = 1024, 640
        self.screen = pygame.Surface((self.width, self.height)) if headless else pygame.display.set_mode((self.width, self.height))
        if not headless:
            pygame.display.set_caption("Robotaxi — Assignment 1")
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 20)
        self.title = pygame.font.Font(None, 34)
        self.clock = pygame.time.Clock()
        self.key = jax.random.PRNGKey(0)
        self.obs, self.state = RobotaxiEnv.reset(self.key, params, initial)
        self.info = {}
        self.running = True

    def point(self, xy):
        h, w = self.params.map_height_width
        cell = self.params.cell_size
        scale = min(920 / ((w - 1) * cell + 2), 430 / ((h - 1) * cell + 2))
        x, y = np.asarray(xy)
        return (int(52 + (x + 1) * scale), int(105 + ((h - 1) * cell + 1 - y) * scale))

    def label(self, text, position, color=(33, 43, 58), small=False):
        self.screen.blit((self.small if small else self.font).render(text, True, color), position)

    def draw(self, map_id=0, mode="Manual drive", episode_return=0.):
        s, p = self.state, self.params
        self.screen.fill((246, 248, 250))
        self.screen.blit(self.title.render("Robotaxi / Environment", True, (20, 35, 55)), (32, 22))
        self.label(f"Assignment 1  |  map {map_id}  |  {mode}", (32, 60))
        scale = self.point(jnp.array([1., 0.]))[0] - self.point(jnp.array([0., 0.]))[0]
        radius = max(2, round(p.radius * scale))
        # A scale-consistent reference grid is useful when inspecting collisions.
        h, w = p.map_height_width
        for row in range(h):
            for col in range(w):
                pygame.draw.circle(self.screen, (218, 224, 230), self.point([col * p.cell_size, row * p.cell_size]), 1)
        for pos in np.asarray(s.static_obstacles):
            pygame.draw.circle(self.screen, (70, 81, 96), self.point(pos), radius)
        path = [self.point(v) for v in np.asarray(s.path_array)]
        if len(path) >= 2:
            pygame.draw.lines(self.screen, (159, 183, 216), False, path, 2)
        center = self.point(s.agent_pos)
        rays_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for ray in np.asarray(self.obs.collision_rays):
            pygame.draw.line(rays_surface, (225, 116, 81, 50), center, self.point(np.asarray(s.agent_pos) + ray), 1)
        self.screen.blit(rays_surface, (0, 0))
        for pos, velocity in zip(np.asarray(s.kinematic_obstacles), np.asarray(s.kinematic_obst_velocities)):
            pt = self.point(pos)
            pygame.draw.circle(self.screen, (212, 86, 61), pt, radius)
            pygame.draw.line(self.screen, (255, 244, 227), pt, self.point(pos + velocity), 2)
        goal = self.point(s.goal_pos)
        pygame.draw.circle(self.screen, (33, 148, 104), goal, max(3, round(p.goal_radius * scale)), 2)
        self.label("GOAL", (goal[0] - 20, goal[1] - 30), (25, 125, 80), True)
        pygame.draw.circle(self.screen, (37, 101, 179), center, radius)
        nose = self.point(np.asarray(s.agent_pos) + p.radius * np.asarray(s.agent_forward_dir))
        pygame.draw.line(self.screen, (255, 230, 119), center, nose, 3)
        pygame.draw.rect(self.screen, (231, 236, 241), (0, 552, self.width, 88))
        status = "RUNNING"
        if bool(s.terminated):
            status = "GOAL REACHED" if bool(self.info.get("goal_reached", False)) else (
                "COLLISION" if bool(self.info.get("collision", False)) else "TIME LIMIT")
        self.label(f"{status}   |   t = {int(s.time) * p.step_size:.1f} s   |   speed = {float(s.speed):.2f} u/s   |   return = {episode_return:.2f}", (32, 568))
        self.label("Blue: taxi (radius 1)    Green: goal    Red: moving obstacles    Grey: static obstacles", (32, 597), small=True)
        if not self.headless:
            pygame.display.flip()

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        throttle = int(keys[pygame.K_UP] or keys[pygame.K_w]) - int(keys[pygame.K_DOWN] or keys[pygame.K_s])
        # Positive steering means left in world coordinates.
        steering = int(keys[pygame.K_LEFT] or keys[pygame.K_a]) - int(keys[pygame.K_RIGHT] or keys[pygame.K_d])
        return jnp.array([throttle, steering], dtype=jnp.float32)

    def frame(self):
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def run(self, map_id=0, demo=False, record=None):
        # Avoid a preexec_fn/fork in the encoder after JAX has started threads.
        os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
        import imageio.v2 as imageio
        if record:
            Path(record).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(record), fps=self.params.fps, codec="libx264", quality=7) if record else None
        episode_return = 0.
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        self.running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        self.obs, self.state = RobotaxiEnv.reset(self.key, self.params, self.init_state)
                        self.info, episode_return = {}, 0.
                if not self.running:
                    break
                if not bool(self.state.terminated):
                    action = demo_action(self.state, self.params) if demo else self.handle_keys()
                    self.obs, self.state, reward, _, self.info = RobotaxiEnv.step(self.key, self.state, action, self.params)
                    episode_return += float(reward)
                self.draw(map_id, "Heuristic controller / no training" if demo else "WASD: drive | R: reset | Esc: exit", episode_return)
                if writer:
                    writer.append_data(self.frame())
                if bool(self.state.terminated) and (record or self.headless):
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
