from datetime import datetime

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

import os

import matplotlib.pyplot as plt

os.environ['CURL_CA_BUNDLE'] = ''

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

data_dir = "data/train/images/"
image = Image.open("images/8056.png")

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# model_path = "detr/checkpoints/detr-r50-e632da11.pth"

# you can specify the revision tag if you don't want the timm dependency
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

def plot_results(pil_img, results):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    for score, label, box, c in zip(results["scores"], results["labels"], results["boxes"], colors):
        clazz = model.config.id2label[label.item()]
        if clazz in {"person", "sports ball"}:
            box = [round(i, 2) for i in box.tolist()]
            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                       fill=False, color=c, linewidth=1))

            text = f'{clazz}: {score:0.2f}'
            # ax.text(xmin, ymin, text, fontsize=8,
            #         bbox=dict(facecolor='yellow', alpha=0.5))

            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

    plt.axis('off')
    plt.show()


# load from local model file
processor = DetrImageProcessor.from_pretrained("./fb-resnet50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("./fb-resnet50", revision="no_timm")

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
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     clazz = model.config.id2label[label.item()]
#     if clazz in {"person", "sports ball"}:
#         box = [round(i, 2) for i in box.tolist()]
#         print(
#             f"Detected {model.config.id2label[label.item()]} with confidence "
#             f"{round(score.item(), 3)} at location {box}"
#         )

# processor.save_pretrained("./fb-resnet50")
# model.save_pretrained("./fb-resnet50")

plot_results(image, results)
