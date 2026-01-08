from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pygame


@dataclass
class PygameViewer:
    width: int = 1280
    height: int = 720
    title: str = "AirWriting Debug"
    bg: Tuple[int, int, int] = (10, 10, 10)

    def __post_init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        self.clock = pygame.time.Clock()
        self.stroke_points: List[Tuple[int, int]] = []
        self.is_running = True

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_running = False
                if event.key == pygame.K_c:
                    # clear strokes
                    self.stroke_points = []

    def draw(self, cursor_xy: Tuple[float, float], down: bool) -> None:
        self.handle_events()
        if not self.is_running:
            return

        self.screen.fill(self.bg)

        x, y = int(cursor_xy[0]), int(cursor_xy[1])
        x = max(0, min(self.width-1, x))
        y = max(0, min(self.height-1, y))

        if down:
            self.stroke_points.append((x, y))

        # draw strokes
        if len(self.stroke_points) >= 2:
            pygame.draw.lines(self.screen, (220, 220, 220), False, self.stroke_points, 2)

        # draw cursor
        color = (0, 220, 120) if down else (220, 120, 0)
        pygame.draw.circle(self.screen, color, (x, y), 6)

        pygame.display.flip()
        self.clock.tick(120)

    def close(self) -> None:
        try:
            pygame.quit()
        except Exception:
            pass
