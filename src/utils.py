import os
import time
from typing import Tuple

import cv2
import numpy as np


class FPSCounter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.last_time = time.time()
        self.fps = 0.0

    def update(self) -> float:
        now = time.time()
        delta = max(now - self.last_time, 1e-6)
        instant = 1.0 / delta
        if self.fps == 0.0:
            self.fps = instant
        else:
            self.fps = (self.smoothing * self.fps) + ((1.0 - self.smoothing) * instant)
        self.last_time = now
        return self.fps


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - h - 6), (x + w + 6, y + 4), bg_color, -1)
    cv2.putText(frame, text, (x + 3, y), font, scale, color, thickness, cv2.LINE_AA)


def compute_circularity(area: float, perimeter: float) -> float:
    if perimeter == 0:
        return 0.0
    return (4.0 * np.pi * area) / (perimeter * perimeter)


def compute_hu_moments(contour: np.ndarray) -> np.ndarray:
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments)
    return hu.flatten()


def scale_contour(contour: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return contour
    return (contour.astype(np.float32) / scale).astype(np.int32)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    text = f"FPS: {fps:.1f}"
    draw_label(frame, text, (10, 25), color=(0, 255, 0), bg_color=(0, 0, 0))
