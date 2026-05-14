import time
from typing import Dict, Tuple

import cv2
import numpy as np

from src.utils import draw_label


class SorterSimulator:
    def __init__(self, config) -> None:
        self.config = config
        self.last_bin = "-"
        self.last_label = "-"
        self.last_update = 0.0
        self.state = "IDLE"

    def update(self, label: str, bin_name: str) -> None:
        self.last_label = label
        self.last_bin = bin_name
        self.last_update = time.time()
        self.state = "SORTING"

    def step(self) -> None:
        if self.state == "SORTING" and (time.time() - self.last_update) > 0.6:
            self.state = "IDLE"

    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        panel_h = 90
        y0 = h - panel_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        draw_label(frame, f"Sorter State: {self.state}", (10, y0 + 25), color=(255, 255, 255))
        draw_label(frame, f"Last: {self.last_label}", (10, y0 + 55), color=(255, 255, 255))

        bin_w = 120
        padding = 10
        x_start = w - (bin_w + padding) * len(self.config.bin_names) - 10
        for idx, bin_name in enumerate(self.config.bin_names):
            x1 = x_start + idx * (bin_w + padding)
            y1 = y0 + 20
            x2 = x1 + bin_w
            y2 = y1 + 50
            is_active = bin_name == self.last_bin and self.state == "SORTING"
            color = (0, 200, 255) if is_active else (60, 60, 60)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            draw_label(frame, bin_name, (x1 + 8, y1 + 30), color=(255, 255, 255), bg_color=(0, 0, 0))


class SortDecision:
    def __init__(self, label: str, bin_name: str, confidence: float) -> None:
        self.label = label
        self.bin_name = bin_name
        self.confidence = confidence

    def to_dict(self) -> Dict[str, str]:
        return {
            "label": self.label,
            "bin": self.bin_name,
            "confidence": f"{self.confidence:.2f}",
        }
