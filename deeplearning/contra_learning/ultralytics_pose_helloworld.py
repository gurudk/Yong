from ultralytics.models import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("/home/wolf/datasets/reid/frames/SR583/SR583_0/SR583_0_0.png", show=True,
                save=True)  # predict on an image

print(results)
