import io
import torch
from PIL import Image
from ultralytics import YOLO

# Model
model = model = YOLO('best.pt')

# img = Image.open("zidane.jpg")  # PIL image direct open

# Read from bytes as we do in app
with open("Oil.jpg", "rb") as file:
    img_bytes = file.read()
img = Image.open(io.BytesIO(img_bytes))

results = model(img, size=640)  # includes NMS

print(results.pandas().xyxy[0])
