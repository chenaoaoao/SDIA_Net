

from ultralytics import YOLO





model = YOLO(r"E:\xiangmu\myIRMA\myyolov8.yaml")  # build a new model from scratch

 
# Train the model
# results = model.train(data="coco130.yaml", epochs=500, imgsz=640)


