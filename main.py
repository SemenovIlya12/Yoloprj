from ultralytics import YOLO
import torch


#model = YOLO("yolov8m.pt")
model = YOLO('absolute.pt')


train_params = {
    "epochs": 20,
    "batch": 16,
    "imgsz": 864,
    "data": "data.yaml",
    "name": "abs",
    "workers" : 4,
    "hsv_h": 0.075,
    "hsv_s": 0.7,
    "hsv_v": 0.3,
    "degrees": 45,
    "translate": 0.2,
    "scale": 0.7,
    "shear": 0.5,
    "perspective": 0.0,
    "flipud": 0.3,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "save_period" : 3
}

model.train(**train_params)

model.save("new_best3.pt")


