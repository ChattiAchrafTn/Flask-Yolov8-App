"""
Run a rest API exposing the yolov8n_custom object detection model
"""
import argparse
import io
from PIL import Image

import torch
from flask import Flask, request

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov8"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=640) 
        return results.pandas().xyxy[0].to_json(orient="records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov8 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', default='yolov8n', help='model to run, i.e. --model yolov8n')
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov8', args.model)
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
