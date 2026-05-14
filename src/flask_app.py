from flask import Flask, Response
import cv2

from src.config import AppConfig
from src.processor import FrameProcessor
from src.video_stream import VideoStream

app = Flask(__name__)

config = AppConfig()
stream = VideoStream(config.camera_index, (config.frame_width, config.frame_height))
processor = FrameProcessor(config)


def frame_generator():
    while True:
        frame = stream.read()
        if frame is None:
            break
        output, _, _ = processor.process(frame)
        ok, buffer = cv2.imencode(".jpg", output)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return "<html><body><h2>Shape & Colour Recognition Sorter</h2><img src='/stream'></body></html>"


@app.route("/stream")
def stream_view():
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
