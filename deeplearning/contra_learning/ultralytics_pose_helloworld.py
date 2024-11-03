from ultralytics.models import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("/home/wolf/datasets/reid/dataset/test_final_dataset_32_128/9/8_SR583_0_111_frame_841_823.png",
                show=True,
                save=True)  # predict on an image

print(results)
