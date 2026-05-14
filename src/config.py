from dataclasses import dataclass


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    process_scale: float = 1.0

    blur_kernel: tuple = (5, 5)
    canny_threshold_1: int = 60
    canny_threshold_2: int = 180
    morph_kernel: tuple = (5, 5)
    morph_iterations: int = 1

    min_contour_area: int = 1500
    approx_epsilon: float = 0.02
    square_aspect_ratio: tuple = (0.9, 1.1)
    circle_circularity: float = 0.78

    min_color_pixels: int = 200

    show_hu_moments: bool = False
    show_side_panel: bool = True
    show_confidence: bool = True

    log_detections: bool = False
    log_path: str = "data/logs/detections.csv"

    screenshot_dir: str = "data/screenshots"

    bin_names: tuple = ("Bin A", "Bin B", "Bin C")
    bin_map: dict = None

    color_ranges: dict = None

    def __post_init__(self) -> None:
        if self.bin_map is None:
            self.bin_map = {
                "Red": "Bin A",
                "Orange": "Bin A",
                "Yellow": "Bin A",
                "Green": "Bin B",
                "Blue": "Bin C",
                "Unknown": "Bin C",
            }

        if self.color_ranges is None:
            self.color_ranges = {
                "Red": [((0, 80, 80), (10, 255, 255)), ((170, 80, 80), (180, 255, 255))],
                "Orange": [((11, 80, 80), (19, 255, 255))],
                "Yellow": [((20, 80, 80), (35, 255, 255))],
                "Green": [((36, 80, 80), (86, 255, 255))],
                "Blue": [((90, 80, 80), (128, 255, 255))],
            }
