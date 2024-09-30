import torchvision
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import os
import json
import time
import smtplib
import datetime

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


def send_mail(SUBJECT, TEXT):
    msg = EmailMessage()
    msg.set_content(TEXT)
    msg['Subject'] = SUBJECT
    msg['From'] = "soccervit@126.com"
    msg['To'] = "soccervit@126.com"

    s = smtplib.SMTP('smtp.126.com')
    s.login('soccervit@126.com', os.environ["auth_code"])
    s.send_message(msg)
    s.quit()


class PlayerDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.annotations = json.loads(f.read())
            self.file_ids = self.annotations["file_ids"]
            self.player_detections_items = [(key, value["target_angle"]) for key, value in
                                            self.annotations["player_detections"].items()]
        self.transform = transform

    def __len__(self):
        return len(self.player_detections_items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.player_detections_items[idx][0]
        image = Image.open(img_name)
        angle = self.player_detections_items[idx][1]
        gt_x = torch.cos(torch.tensor(angle))
        gt_y = torch.sin(torch.tensor(angle))

        truth_tensor = torch.tensor([gt_x, gt_y]).squeeze()

        sample = {'image': image, 'ground_truth': truth_tensor, 'image_name': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


def cos_similiar(y_pred, y):
    return 2 - 2 * torch.cos(y_pred - y)


def loss_func(y_pred, y):
    # angle loss
    x0_pred = y_pred[:, 0]
    y0_pred = y_pred[:, 1]
    theta_ypred = torch.arctan2(y0_pred, x0_pred)
    theta_ygt = torch.arctan2(y[:, 1], y[:, 0])
    theta_loss = 2 - 2 * torch.cos(theta_ypred - theta_ygt)

    distance_loss = torch.abs(torch.sqrt(x0_pred ** 2 + y0_pred ** 2) - 1)

    return theta_loss + distance_loss


def theta_loss(y_pred, y):
    x0_pred = y_pred[:, 0]
    y0_pred = y_pred[:, 1]
    theta_ypred = torch.arctan2(y0_pred, x0_pred)
    theta_ygt = torch.arctan2(y[:, 1], y[:, 0])
    theta_loss = 2 - 2 * torch.cos(theta_ypred - theta_ygt)

    return theta_loss


def distance_loss(y_pred, y):
    # angle loss
    x0_pred = y_pred[:, 0]
    y0_pred = y_pred[:, 1]
    distance_loss = torch.abs(torch.sqrt(x0_pred ** 2 + y0_pred ** 2) - 1)

    return distance_loss


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
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
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
                 dim_head=64, out_dim=4, dropout=0., emb_dropout=0.):
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
        return self.mlp(x).sigmoid()


LR = 1e-5
TOTAL_EPOCHS = 1000
BATCH_SIZE = 64
AVG_BATCH_SIZE = 50
full_dataset_file = "./player_annotation/clean_body_orientation_atan2_mergeall_07.json.20240928093915"

LOG_FILE = "./log/train.log." + get_nowtime_str()
validation_rate = .1
shuffle_dataset = True
random_seed = 0

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((48, 96)),
    v2.ToTensor(),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_dataset = PlayerDataset(full_dataset_file, transform)
full_size = len(full_dataset)
indices = list(range(full_size))
split = int(np.floor(validation_rate * full_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# print("train samples:", len(train_dataset), "val samples:", len(val_dataset))

train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# --------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Set device GPU or CPU where the training will take place

# -------------------------Create model -------------------------------------------

model = PlayerBodyOrientationViT(
    image_size=(48, 96),
    patch_size=(8, 16),
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)

model = model.to(device)
# -------------------------Create model -------------------------------------------


optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)  # Create adam optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    'min',
    factor=0.3,
    patience=20,
    verbose=True,
    min_lr=1e-8,
)

# ----------------Train--------------------------------------------------------------------------

for epoch in trange(TOTAL_EPOCHS + 1, desc="Training.."):  # Training loop
    train_loss = 0.0
    train_theta_loss = 0
    train_distance_loss = 0
    start = time.time()

    model.train()

    for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
    ):
        x = batch['image']
        y = batch['ground_truth']

        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        train_th_loss = theta_loss(y_pred, y).sum()
        train_dis_loss = distance_loss(y_pred, y).sum()

        loss = train_th_loss

        train_loss += loss.detach().cpu().item() / len(train_indices)
        train_theta_loss += train_th_loss.detach().cpu().item() / len(train_indices)
        train_distance_loss += train_dis_loss.detach().cpu().item() / len(train_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    duration = time.time() - start

    if epoch % 20 == 0:  # Save model
        print("Saving Model" + str(epoch) + ".torch")  # Save model weight
        torch.save(model.state_dict(), "./zoo/player_resnet18_" + str(epoch) + ".torch")

    # ------------------------------------valiadate step----------------------------------------
    print("-----------------------------validate step--------------------------")

    val_loss = 0
    val_theta_loss = 0
    val_distance_loss = 0

    model.eval()

    with torch.no_grad():
        idx = 0
        for batch in val_loader:
            x = batch['image']
            y = batch['ground_truth']

            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            val_th_loss = theta_loss(y_pred, y).sum()
            val_dis_loss = distance_loss(y_pred, y).sum()

            loss = val_th_loss

            val_loss += loss.detach().cpu().item() / len(val_indices)
            val_theta_loss += val_th_loss.detach().cpu().item() / len(val_indices)
            val_distance_loss += val_dis_loss.detach().cpu().item() / len(val_indices)

            idx += 1

    # summary = f"[{epoch}/{TOTAL_EPOCHS}],train loss: {train_loss:.4f} theta train loss is {train_theta_loss:.4f} and distance train loss is {train_distance_loss:.4f}, val loss: {val_loss:.4f} theta val loss is {val_theta_loss:.4f} and distance val loss is {val_distance_loss:.4f},"
    summary = f"[{epoch}/{TOTAL_EPOCHS}],train loss: {train_loss:.4f} , val loss: {val_loss:.4f} "
    with open(LOG_FILE, 'a') as log:
        log.write(summary + '\n')
    print(summary)
    # nowtime = datetime.datetime.now()
    # try:
    #     send_mail("[" + nowtime.strftime("%Y%m%d") + "]-" + summary, 'ATT..............................')
    # except Exception:
    #     print("Sending email failed!")
    #     print(traceback.format_exc())

    scheduler.step(val_loss)
