import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':

    from ultralytics import YOLO

    # Create a new YOLO model from scratch
    model = YOLO(r"E:\xiangmu\myIRMA\myyolov8.yaml")

    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO(r'E:\xiangmu\ultralytics11.27\yolov8n.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data=r'E:\xiangmu\myIRMA\mycoco128.yaml', epochs=1, imgsz=640,batch=8)

    # Evaluate the model's performance on the validation set
    results = model.val()


    # Perform object detection on an image using the model
   # results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
  #  success = model.export(format='onnx')
