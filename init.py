import os
import pygame as pg

COLORS = {
    "background": (0, 0, 1),
    "white": (255, 255, 255),
    "key": (0, 0, 1),
    "static": (99, 99, 99)
}

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 100)
aspect_ratio = 4 / 4  # 9/16
WIDTH = 300  # 1280  # 1280 560
HEIGHT = int(WIDTH * aspect_ratio)
SCREEN = pg.display.set_mode((WIDTH, HEIGHT))
BACKGROUND = pg.image.load("res/map-blueprint.jpg")
BACKGROUND = pg.transform.scale(BACKGROUND, (WIDTH, HEIGHT))

FPS = 10

NUM_AGENTS = 5
