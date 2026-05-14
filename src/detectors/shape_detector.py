from typing import Tuple

import cv2
import numpy as np

from src.utils import compute_circularity, compute_hu_moments


def find_contours(edge_frame: np.ndarray, min_area: int) -> list:
    contours, _ = cv2.findContours(edge_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def classify_shape(
    contour: np.ndarray,
    approx_epsilon: float,
    square_aspect_ratio: Tuple[float, float],
    circle_circularity: float,
) -> dict:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, approx_epsilon * perimeter, True)
    vertices = len(approx)
    area = cv2.contourArea(contour)
    circularity = compute_circularity(area, perimeter)

    shape = "Unknown"
    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h) if h > 0 else 0.0
        if square_aspect_ratio[0] <= aspect_ratio <= square_aspect_ratio[1]:
            shape = "Square"
        else:
            shape = "Rectangle"
    else:
        if circularity >= circle_circularity:
            shape = "Circle"
        elif vertices > 4:
            shape = "Polygon"

    hu = compute_hu_moments(contour)

    return {
        "shape": shape,
        "approx": approx,
        "vertices": vertices,
        "circularity": circularity,
        "hu": hu,
    }
