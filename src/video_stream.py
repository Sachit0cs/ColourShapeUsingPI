from typing import Optional, Tuple

import cv2


class VideoStream:
    def __init__(self, camera_index: int, frame_size: Tuple[int, int]) -> None:
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    def read(self) -> Optional[cv2.Mat]:
        if not self.capture.isOpened():
            return None
        ok, frame = self.capture.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
