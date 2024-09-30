import torch
import matplotlib.pyplot as plt

import cv2

import datetime
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

import torchvision
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import os
import json
import time
import smtplib
import datetime
import traceback

from email.message import EmailMessage

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch import nn
from einops import rearrange, repeat

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from torchvision.transforms import v2
from einops.layers.torch import Rearrange

np.random.seed(0)
torch.manual_seed(0)


def get_nowtime_str():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d%H%M%S")


def get_detection_results(image, model, processor, threshold=0.5):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        start_time = datetime.datetime.now().timestamp()
        outputs = model(**inputs)
        end_time = datetime.datetime.now().timestamp()
        print("model execution time:", end_time - start_time)

    results = processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]),
                                                      threshold=threshold)
    return results


def plot_results(pil_img, results, model):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    player_idx = 0
    for result in results:
        for score, label, box, c in zip(result["scores"], result["labels"], result["boxes"], colors):
            clazz = model.config.id2label[label.item()]
            if clazz in {"person"}:
                box = [round(i, 2) for i in box.tolist()]
                xmin, ymin, xmax, ymax = box
                ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                           fill=False, color=c, linewidth=1))

                text = f'{clazz} {player_idx}: {score:0.2f}'
                ax.text(xmin, ymin, text, fontsize=8,
                        bbox=dict(facecolor='yellow', alpha=0.5))

                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

                player_idx += 1

    plt.axis('off')
    plt.show()


def get_nowtime_str():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d%H%M%S")


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.leaky_relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class PlayerBodyOrientationViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, out_dim=2, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(num_patches + 1, dim),
            persistent=False,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp = MLP(dim, dim, out_dim, 3)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        x += self.positional_embeddings.repeat(b, 1, 1)

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp(x)


def get_target_point(target_box):
    xmin, ymin, xmax, ymax = target_box.tolist()

    return int(xmin + xmax) // 2, int(ymin)


def get_angle_from_xy(xy_tensor):
    x0 = xy_tensor[:, 0]
    y0 = xy_tensor[:, 1]
    theta_angle = torch.arctan2(y0, x0)
    return theta_angle


def get_plain_value(tensor_value):
    return tensor_value.detach().cpu().item()


def get_cross_point_with_top(input_angle, x0, y0, width, height):
    k = np.tan(input_angle)
    if k > height or k < -height:
        a = np.array([[0, 1], [1, 0]])
        b = np.array([0, x0])
    else:
        a = np.array([[0, 1], [-k, 1]])
        b = np.array([0, y0 - k * x0])

    x = np.linalg.solve(a, b)
    return x[0], x[1]


def get_cross_point_with_down(input_angle, x0, y0, width, height):
    k = np.tan(input_angle)
    if k > height or k < -height:
        a = np.array([[0, 1], [1, 0]])
        b = np.array([height, x0])
    else:
        a = np.array([[0, 1], [-k, 1]])
        b = np.array([height, y0 - k * x0])

    x = np.linalg.solve(a, b)
    return x[0], x[1]


def get_cross_point_with_left(input_angle, x0, y0, width, height):
    k = np.tan(input_angle)
    if k > height or k < -height:
        a = np.array([[1, 0], [0, 1]])
        b = np.array([0, y0])
    else:
        a = np.array([[1, 0], [-k, 1]])
        b = np.array([0, y0 - k * x0])

    x = np.linalg.solve(a, b)
    return x[0], x[1]


def get_cross_point_with_right(input_angle, x0, y0, width, height):
    # right
    k = np.tan(input_angle)
    if k > height or k < -height:
        a = np.array([[1, 0], [0, 1]])
        b = np.array([width, y0])
    else:
        a = np.array([[1, 0], [-k, 1]])
        b = np.array([width, y0 - k * x0])

    x = np.linalg.solve(a, b)
    return x[0], x[1]


def get_view_range_polygons(base_point, angle, delta_angle, height=720, width=1280):
    polygons = []

    theta_start = angle - delta_angle
    theta_end = angle + delta_angle

    x0, y0 = base_point
    polygons.append([x0, y0])
    thetas = [np.arctan2(-y0, -x0), np.arctan2(-y0, width - x0), np.arctan2(height - y0, width - x0),
              np.arctan2(height - y0, -x0)]

    # -------------------if start angle in top -------------------------------------
    if thetas[1] > theta_start >= thetas[0]:
        x = get_cross_point_with_top(theta_start, x0, y0, width, height)
        polygons.append([x[0], x[1]])

        # deal end
        if thetas[1] > theta_end >= thetas[0]:
            x = get_cross_point_with_top(theta_end, x0, y0, width, height)
            polygons.append([x[0], x[1]])
        elif thetas[2] > theta_end >= thetas[1]:
            x = get_cross_point_with_right(theta_end, x0, y0, width, height)
            polygons.append([width, 0])
            polygons.append([x[0], x[1]])
            # three points
        elif thetas[3] > theta_end >= thetas[2]:
            x = get_cross_point_with_down(theta_end, x0, y0, width, height)
            polygons.append([width, 0])
            polygons.append([width, height])
            polygons.append([x[0], x[1]])
            # four points

    # -------------------if start angle in right -------------------------------------
    if thetas[2] > theta_start >= thetas[1]:
        x = get_cross_point_with_right(theta_start, x0, y0, width, height)
        polygons.append([x[0], x[1]])

        # deal end
        if thetas[2] > theta_end >= thetas[1]:
            x = get_cross_point_with_right(theta_end, x0, y0, width, height)
            polygons.append([x[0], x[1]])
        elif thetas[3] > theta_end >= thetas[2]:
            x = get_cross_point_with_down(theta_end, x0, y0, width, height)
            polygons.append([width, height])
            polygons.append([x[0], x[1]])
            # three points
        elif theta_end >= thetas[3]:
            x = get_cross_point_with_left(theta_end, x0, y0, width, height)
            polygons.append([0, height])
            polygons.append([width, height])
            polygons.append([x[0], x[1]])
            # four points

    # -------------------if start angle in down -------------------------------------
    if thetas[3] > theta_start >= thetas[2]:
        x = get_cross_point_with_down(theta_start, x0, y0, width, height)
        polygons.append([x[0], x[1]])

        # deal end
        if thetas[3] > theta_end >= thetas[2]:
            x = get_cross_point_with_down(theta_end, x0, y0, width, height)
            polygons.append([x[0], x[1]])
        elif theta_end >= thetas[3] or theta_end < thetas[0]:
            x = get_cross_point_with_left(theta_end, x0, y0, width, height)
            polygons.append([0, height])
            polygons.append([x[0], x[1]])
            # three points
        elif theta_end >= thetas[3]:
            x = get_cross_point_with_top(theta_end, x0, y0, width, height)
            polygons.append([0, height])
            polygons.append([0, 0])
            polygons.append([x[0], x[1]])
            # four points

    # -------------------if start angle in left -------------------------------------
    if theta_start >= thetas[3] or theta_start < thetas[0]:
        x = get_cross_point_with_left(theta_start, x0, y0, width, height)
        polygons.append([x[0], x[1]])

        # deal end
        if theta_end >= thetas[3] or theta_end < thetas[0]:
            x = get_cross_point_with_left(theta_end, x0, y0, width, height)
            polygons.append([x[0], x[1]])
        elif thetas[1] > theta_end >= thetas[0]:
            x = get_cross_point_with_left(theta_end, x0, y0, width, height)
            polygons.append([0, 0])
            polygons.append([x[0], x[1]])
            # three points
        elif thetas[2] > theta_end >= thetas[1]:
            x = get_cross_point_with_top(theta_end, x0, y0, width, height)
            polygons.append([0, 0])
            polygons.append([width, 0])
            polygons.append([x[0], x[1]])
            # four points

    return np.int32(polygons)


def draw_view_range_polygon(img, base_point, angle, delta_angle, height, width, alpha=0.2):
    # Create a blank image (black background)
    # img = np.zeros((512, 512, 3), np.uint8)
    # img = cv2.imread("videos/frames/B1606b0e6_1 (34)/B1606b0e6_1 (34)_23.png")

    # Define the polygon vertices
    pts = get_view_range_polygons(base_point, angle, delta_angle, height, width)

    pts = pts.reshape((-1, 1, 2))

    # result = cv2.fillPoly(img, [pts], (255, 0, 0))

    # Create a mask for the polygon
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    # Create an overlay image with the desired color and transparency
    overlay = np.zeros_like(img)
    cv2.fillPoly(overlay, [pts], (0, 0, 255))  # Green color

    # Apply the overlay to the original image using the mask
    result = cv2.addWeighted(overlay, alpha, img, 1, 0, img)

    return result


def process_image_frame(cv2image):
    color_coverted = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(color_coverted)
    img_height, img_width, channel = cv2image.shape
    results = get_detection_results(image, detr_model, detr_processor, threshold=0.7)

    player_idx = 0
    for result in results:
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            clazz = detr_model.config.id2label[label.item()]
            target_point = get_target_point(box)

            if clazz in {"person"}:
                (xmin, ymin, xmax, ymax) = box.tolist()

                crop = image.crop((xmin - delta, ymin - delta, xmax + delta, ymax + delta))
                player_tensor = transform(crop).unsqueeze(0)
                player_tensor = player_tensor.to(device)
                angle_pred = player_orientation_model(player_tensor)

                theta = get_plain_value(get_angle_from_xy(angle_pred))

                cv2image = draw_view_range_polygon(cv2image, target_point, theta, delta_angle, img_height, img_width,
                                                   alpha=0.1)

                print("person ", player_idx, "'s angle:", theta)

                player_idx += 1
                if player_idx == 8:
                    break
    return cv2image


# ------------------------constans define------------------------------------
LOCAL_MODEL_DIR = "./rtdetr_r50vd"
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

LR = 1e-5
TOTAL_EPOCHS = 2000
BATCH_SIZE = 1
AVG_BATCH_SIZE = 50
full_dataset_file = "./player_annotation/clean_body_orientation_loss255555_mergeall_val_07.json.20240928093915"
test_model_file = "./model/cleandata_loss15_16x16_160.torch"
LOG_FILE = "./log/train.log." + get_nowtime_str()
player_body_orientation_data_explored_file = "./explored/loss2555555_val_data_explored_file_140pth_angle.txt." + get_nowtime_str()
validation_rate = 0
shuffle_dataset = True
random_seed = 0

delta = 5
player_angle_range = np.pi / 6

# ------------------------constans define------------------------------------

# --------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Set device GPU or CPU where the training will take place

# ------------------------- Model define-------------------------------------------

player_orientation_model = PlayerBodyOrientationViT(
    image_size=(48, 96),
    patch_size=(3, 6),
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)

player_orientation_model = player_orientation_model.to(device)
player_orientation_model.load_state_dict(torch.load(test_model_file))

# init processor and model
detr_processor = RTDetrImageProcessor.from_pretrained(LOCAL_MODEL_DIR)
detr_model = RTDetrForObjectDetection.from_pretrained(LOCAL_MODEL_DIR)

# ------------------------- Model define-------------------------------------------


transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((48, 96)),
    v2.ToTensor(),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_file_name = "videos/frames/C35bd9041_0 (19)/C35bd9041_0 (19)_228.png"
# test_file_name = "./images/5403.png"
delta_angle = np.pi / 12
my_video = "./videos/A1606b0e6_0 (63).mp4"

cap = cv2.VideoCapture(my_video)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
out = cv2.VideoWriter('./videos/A1606b0e6_0 (63)_processed.mp4', fourcc, fps, (width, height))

frame_index = 0
# Read and write frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    print(frame_index)
    # Process the frame (optional)

    frame = process_image_frame(frame)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (optional)
    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    frame_index += 1

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

# cv2image = cv2.imread(test_file_name)
# cv2image = process_image_frame(cv2image)
# cv2.imwrite("test_poly.jpg", cv2image)
# plot_results(image, results, detr_model)
