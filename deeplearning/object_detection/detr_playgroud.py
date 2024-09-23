from datetime import datetime

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

import os

os.environ['CURL_CA_BUNDLE'] = ''

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

access_token = "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"

data_dir = "data/train/images/"
test_image = data_dir + "6_360x640.png"
image = Image.open(test_image)

# model_path = "detr/checkpoints/detr-r50-e632da11.pth"

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm", token=access_token)
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm", token=access_token)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
inputs = processor(images=image, return_tensors="pt").to(device)
model = model.to(device)

start_time = datetime.now().timestamp()
outputs = model(**inputs)
end_time = datetime.now().timestamp()
print("model execution time:", end_time - start_time)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
