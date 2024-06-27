import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import time
import math
import random
import os

TRAINED_CKPT_PATH = 'trained_weights/checkpoint.pth'
checkpoint = torch.load(TRAINED_CKPT_PATH, map_location='cpu')
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=2)
model.load_state_dict(checkpoint['model'], strict=False)

CLASSES = ['face']

# colors for visualization
COLORS = [[0.000, 0.447, 0.741]]

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
    plt.close()


def postprocess_img(img_path):
    im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    start = time.time()
    outputs = model(img)
    end = time.time()
    print(f'Prediction time per image: {end - start} ', )
    # print(f'Prediction time per image: {math.ceil(end - start)}s ', )

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    plot_results(im, probas[keep], bboxes_scaled)


TEST_IMG_PATH1 = 'datasets/WIDER_test/images'
TEST_IMG_PATH = 'datasets/mytest'

img_format = {'jpg', 'png', 'jpeg'}
paths = list()

for obj in os.scandir(TEST_IMG_PATH):
    # paths_temp = [obj.path for obj in os.scandir(obj.path)]
    paths.append(obj.path)

# for obj in os.scandir(TEST_IMG_PATH1):
#     if obj.is_dir():
#         paths_temp = [obj.path for obj in os.scandir(obj.path) if obj.name.split(".")[-1] in img_format]
#         paths.extend(paths_temp)

print('Total number of test images: ', len(paths))
random.shuffle(paths)

for i in paths[1:15]:
    postprocess_img(i)
