import time

import cv2

from src.config import AppConfig
from src.processor import FrameProcessor
from src.utils import ensure_dir
from src.video_stream import VideoStream


def main() -> None:
    config = AppConfig()
    stream = VideoStream(config.camera_index, (config.frame_width, config.frame_height))
    processor = FrameProcessor(config)

    while True:
        frame = stream.read()
        if frame is None:
            print("Failed to read from webcam.")
            break

        output, _, _ = processor.process(frame)
        cv2.imshow("Shape & Colour Recognition Sorter", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ensure_dir(config.screenshot_dir)
            filename = f"{config.screenshot_dir}/snapshot_{int(time.time())}.png"
            cv2.imwrite(filename, output)
            print(f"Saved {filename}")

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
