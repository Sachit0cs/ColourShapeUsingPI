from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.analytics.logger import CSVLogger
from src.config import AppConfig
from src.detectors.color_detector import classify_color
from src.detectors.shape_detector import classify_shape, find_contours
from src.simulation.sorter_sim import SortDecision, SorterSimulator
from src.utils import FPSCounter, compute_hu_moments, draw_fps, draw_label, scale_contour


class FrameProcessor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.sorter = SorterSimulator(config)
        self.fps_counter = FPSCounter()
        self.logger = CSVLogger(config.log_path) if config.log_detections else None
        self.counts: Dict[str, int] = {}

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        scale = self.config.process_scale
        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.config.blur_kernel, 0)
        edges = cv2.Canny(blur, self.config.canny_threshold_1, self.config.canny_threshold_2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.morph_kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=self.config.morph_iterations)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return frame, edges, hsv, scale

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, str]], float]:
        work_frame, edges, hsv, scale = self.preprocess(frame)
        contours = find_contours(edges, self.config.min_contour_area)

        detections: List[Dict[str, str]] = []
        best_decision = None
        best_area = 0.0

        for contour in contours:
            shape_info = classify_shape(
                contour,
                self.config.approx_epsilon,
                self.config.square_aspect_ratio,
                self.config.circle_circularity,
            )
            shape_name = shape_info["shape"]

            color_name, color_conf = classify_color(
                hsv, contour, self.config.color_ranges, self.config.min_color_pixels
            )
            label = f"{color_name} {shape_name}"
            bin_name = self.config.bin_map.get(color_name, "Bin C")

            area = cv2.contourArea(contour)
            if area > best_area:
                best_area = area
                best_decision = SortDecision(label, bin_name, color_conf)

            scaled_contour = scale_contour(contour, scale)
            cv2.drawContours(frame, [scaled_contour], -1, (0, 255, 255), 2)

            x, y, w, h = cv2.boundingRect(scaled_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

            confidence_text = f"{color_conf:.2f}" if self.config.show_confidence else ""
            draw_label(frame, label, (x, max(15, y - 8)), color=(255, 255, 255), bg_color=(0, 0, 0))
            if confidence_text:
                draw_label(frame, f"Conf: {confidence_text}", (x, y + h + 18), color=(200, 255, 200))

            if self.config.show_hu_moments:
                hu = compute_hu_moments(contour)
                draw_label(frame, f"Hu1: {hu[0]:.2e}", (x, y + h + 38), color=(200, 200, 255))

            detections.append(
                {
                    "label": label,
                    "shape": shape_name,
                    "color": color_name,
                    "bin": bin_name,
                    "confidence": f"{color_conf:.2f}",
                }
            )

            self.counts[label] = self.counts.get(label, 0) + 1

        if best_decision:
            self.sorter.update(best_decision.label, best_decision.bin_name)
            if self.logger:
                self.logger.log(
                    {
                        "label": best_decision.label,
                        "shape": best_decision.label.split(" ", 1)[-1],
                        "color": best_decision.label.split(" ", 1)[0],
                        "bin": best_decision.bin_name,
                        "confidence": f"{best_decision.confidence:.2f}",
                    }
                )

        self.sorter.step()
        self.sorter.draw(frame)

        if self.config.show_side_panel:
            self._draw_side_panel(frame)

        fps = self.fps_counter.update()
        draw_fps(frame, fps)

        return frame, detections, fps

    def _draw_side_panel(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        panel_w = 220
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        draw_label(frame, "Detections", (w - panel_w + 10, 25), color=(255, 255, 255))
        y = 50
        for label, count in sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)[:8]:
            draw_label(frame, f"{label}: {count}", (w - panel_w + 10, y), color=(200, 255, 200))
            y += 22
