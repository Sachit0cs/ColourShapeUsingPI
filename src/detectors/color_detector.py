from typing import Dict, List, Tuple

import cv2
import numpy as np


def classify_color(
    hsv_frame: np.ndarray,
    contour: np.ndarray,
    color_ranges: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]],
    min_color_pixels: int,
) -> Tuple[str, float]:
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    best_color = "Unknown"
    best_score = 0

    for name, ranges in color_ranges.items():
        combined = np.zeros_like(mask)
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            range_mask = cv2.inRange(hsv_frame, lower_np, upper_np)
            combined = cv2.bitwise_or(combined, range_mask)

        masked = cv2.bitwise_and(combined, combined, mask=mask)
        score = cv2.countNonZero(masked)
        if score > best_score:
            best_score = score
            best_color = name

    total = cv2.countNonZero(mask)
    if best_score < min_color_pixels or total == 0:
        return "Unknown", 0.0

    confidence = best_score / float(total)
    return best_color, confidence
