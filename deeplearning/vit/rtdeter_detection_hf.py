import torch
import requests
import matplotlib.pyplot as plt

from datetime import datetime
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


def get_detection_results(image, model, processor, threshold=0.5):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        start_time = datetime.now().timestamp()
        outputs = model(**inputs)
        end_time = datetime.now().timestamp()
        print("model execution time:", end_time - start_time)

    results = processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]),
                                                      threshold=threshold)
    return results


LOCAL_MODEL_DIR = "./rtdetr_r50vd"

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
# model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")


def plot_results(pil_img, results):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for result in results:
        for score, label, box, c in zip(result["scores"], result["labels"], result["boxes"], colors):
            clazz = model.config.id2label[label.item()]
            if clazz in {"person", "sports ball"}:
                box = [round(i, 2) for i in box.tolist()]
                xmin, ymin, xmax, ymax = box
                ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                           fill=False, color=c, linewidth=1))

                text = f'{clazz}: {score:0.2f}'
                ax.text(xmin, ymin, text, fontsize=8,
                        bbox=dict(facecolor='yellow', alpha=0.5))

                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

    plt.axis('off')
    plt.show()

    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)


# init processor and model
processor = RTDetrImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
model = RTDetrForObjectDetection.from_pretrained(LOCAL_MODEL_DIR)

image = Image.open("videos/frames/B1606b0e6_1 (34)/B1606b0e6_1 (34)_150.png")
results = get_detection_results(image, model, processor, threshold=0.6)

# for result in results:
#     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
#         score, label = score.item(), label_id.item()
#         box = [round(i, 2) for i in box.tolist()]
#         print(f"{model.config.id2label[label]}: {score:.2f} {box}")

plot_results(image, results)

# processor.save_pretrained("./rtdetr_r50vd")
# model.save_pretrained("./rtdetr_r50vd")
