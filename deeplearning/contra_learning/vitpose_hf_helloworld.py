from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
import matplotlib.pyplot as plt
from PIL import Image
import requests

url_image = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url_image, stream=True).raw)

# initialize the model and processor
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

# infer
inputs = processor(image, return_tensors="pt").to(model.device, model.dtype)
outputs = model(**inputs)

# postprocess
image_sizes = [(image.size[1], image.size[0])]
outputs = processor.post_process_keypoint_detection(outputs, image_sizes)
keypoints = outputs[0]["keypoints"].detach().numpy()
scores = outputs[0]["scores"].detach().numpy()
image_width, image_height = image.size

# plot
plt.axis('off')
plt.imshow(image)
plt.scatter(
    keypoints[:, 0],
    keypoints[:, 1],
    s=scores * 100,
    c='cyan',
    alpha=0.4
)
plt.show()
