from init import *
import pygame as pg
# import pygame.gfxdraw
import numpy as np

AGENT_SIZE = 5
AGENT_VIEW_RADIUS = 10 * AGENT_SIZE
AGENT_IMG = pg.Surface((2 * AGENT_SIZE, 2 * AGENT_SIZE), pg.SRCALPHA)
# draw.circle is not anti-aliased and looks rather ugly.
# pg.draw.circle(AGENT_IMG, (0, 255, 0), (1, 1), 1)
# gfxdraw.aacircle looks a bit better.
# pg.gfxdraw.aacircle(AGENT_IMG, 15, 15, 15, (0, 255, 0))
# pg.gfxdraw.filled_circle(AGENT_IMG, 15, 15, 15, (0, 255, 0))
TRANSITION_DELAY = FPS


class Agent(pg.sprite.Sprite):
    def __init__(self, id_num=0, location=None, color=(0, 255, 0)):
        super().__init__()
        self.id = id_num
        if location is None:
            location = [WIDTH // 2, HEIGHT // 2]
        self.image = AGENT_IMG.copy()
        pg.draw.circle(self.image, color, (AGENT_SIZE, AGENT_SIZE), AGENT_SIZE)
        self.location = location
        self.location_delta = 0
        self.rect = self.image.get_rect(center=tuple(self.location))
        self.direction = 0
        self.direction_delta = 0
        #
        self.r_mean = 5
        self.theta_mean = 0
        #
        self.time = TRANSITION_DELAY
        #
        self.view = pg.Rect(0, 0, 2 * AGENT_VIEW_RADIUS, 2 * AGENT_VIEW_RADIUS)
        self.view.center = self.rect.center

    def update(self):
        if self.time < TRANSITION_DELAY:
            self.time += 1
        else:
            self.location_delta = np.random.exponential(self.r_mean, 1)
            self.direction_delta = np.random.uniform(0, np.pi, 1) - np.pi / 2
            self.time = 0

        self.direction += self.direction_delta * (self.time + 1) / TRANSITION_DELAY
        this_r = self.location_delta * (self.time + 1) / TRANSITION_DELAY

        self.location = [
            self.location[0] + int(this_r * np.cos(self.direction)),
            self.location[1] + int(this_r * np.sin(self.direction))
        ]

        self.rect.center = tuple(self.location)
        self.view.center = self.rect.center
        #
        if not (0 <= self.location[0] < WIDTH or 0 <= self.location[0] < HEIGHT):
            self.kill()

    def snapshot(self, t=0):
        screenshot = pg.Surface((2 * AGENT_VIEW_RADIUS, 2 * AGENT_VIEW_RADIUS))
        screenshot.blit(SCREEN, (0, 0), area=self.view)
        pg.image.save(screenshot, "snap/" + str(self.id) + "_" + str(t).zfill(3) + ".jpg")
