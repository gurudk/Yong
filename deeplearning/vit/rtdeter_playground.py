import torch
import requests

from datetime import datetime
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("images/final_dataset_train_94.png")

image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    start_time = datetime.now().timestamp()
    outputs = model(**inputs)
    end_time = datetime.now().timestamp()
    print("model execution time:", end_time - start_time)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]),
                                                        threshold=0.3)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
