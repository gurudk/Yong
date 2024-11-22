import torchvision
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import os
import re
import glob
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


class PlayerClassifyDataset(Dataset):
    def __init__(self, player_json_file, transform=None):
        with open(player_json_file, 'r') as rf:
            self.player_dataset = json.loads(rf.read())
            self.player_json = self.player_dataset["player_list"]
            self.player_dict = self.player_dataset["player_dict"]
            self.player_items = list(self.player_json.items())
        self.transform = transform

    def __len__(self):
        return len(self.player_json)

    def get_class_num(self):
        return len(self.player_dict)

    def __getitem__(self, idx):
        image_name = self.player_items[idx][0]
        ground_truth = self.player_items[idx][1]
        img = Image.open(image_name)

        sample = {'image_tensor': img, 'ground_truth': ground_truth}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_nowtime_str():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d%H%M%S")


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ================================ ViT DEFINE ===================================

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


class PlayerClassifyViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, out_dim=256, dropout=0., emb_dropout=0.):
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

        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim)
        )

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
        return self.mlp(x), x


def get_latest_model_file(model_dir):
    files = list(filter(os.path.isfile, glob.glob(model_dir + "*")))

    if len(files) == 0:
        return (None, 0)
    else:
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_file = files[0]
        latest_epoch = int(re.split(r"[_\.]", latest_file)[-2])
        return latest_file, latest_epoch


# ========================= Training =========================================================

LR = 1e-5
TOTAL_EPOCHS = 500
BATCH_SIZE = 128
full_dataset_file = "/home/wolf/datasets/reid/dataset/classify/train_classify_minnum20_span3.json.20241122105138"

LOG_FILE = "./log/vit_player_classify/vit_classify_final.log." + get_nowtime_str()
SOFTMAX_LOG_FILE = "./log/vit_player_classify/vit_classify_final_softmax.log." + get_nowtime_str()
validation_rate = .1
shuffle_dataset = True
random_seed = 0

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((48, 96)),
    v2.ToTensor(),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_dataset = PlayerClassifyDataset(full_dataset_file, transform)
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
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device(
    'cpu')  # Set device GPU or CPU where the training will take place

# -------------------------Create model -------------------------------------------


model = PlayerClassifyViT(
    image_size=(48, 96),
    patch_size=(3, 6),
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1,
    out_dim=full_dataset.get_class_num()
)

model = model.to(device)

# read the resume weights
latest_model_file, latest_epoch = get_latest_model_file("./zoo/vit_player_classify/")
if latest_model_file:
    model.load_state_dict(torch.load(latest_model_file))

# -------------------------Create model -------------------------------------------

optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)  # Create adam optimizer
criterion = nn.CrossEntropyLoss(reduction="sum")
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    'min',
    factor=0.3,
    patience=20,
    verbose=True,
    min_lr=1e-8,
)

# ----------------Train--------------------------------------------------------------------------

len_datasize = len(train_sampler)

for epoch in trange(TOTAL_EPOCHS + 1, desc="Training.."):  # Training loop

    # ------------------------------------train step----------------------------------------
    train_loss = 0.0
    start = time.time()

    model.train()

    for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
    ):
        x_batch = batch['image_tensor'].squeeze()
        x_truth = batch['ground_truth'].squeeze()
        x_batch = x_batch.to(device)
        x_truth = x_truth.to(device)

        y_pre, y_features = model(x_batch)

        loss = criterion(y_pre, x_truth)

        train_loss += loss.detach().cpu().item() / len(train_indices)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ------------------------------------valiadate step----------------------------------------
    print("-----------------------------validate step--------------------------")

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x_batch = batch['image_tensor'].squeeze()
            x_truth = batch['ground_truth'].squeeze()
            x_batch = x_batch.to(device)
            x_truth = x_truth.to(device)

            y_pre, _ = model(x_batch)

            loss = criterion(y_pre, x_truth)

            val_loss += loss.detach().cpu().item() / len(val_indices)

    scheduler.step(val_loss)

    # ------------------------------------print summary step----------------------------------------
    duration = time.time() - start
    summary = f"[{latest_epoch}/{TOTAL_EPOCHS}] train loss: {train_loss:.4f} , val loss: {val_loss:.4f} , duration:{duration:.2f}, lr:{scheduler.get_last_lr()}"
    with open(LOG_FILE, 'a') as log:
        log.write(summary + '\n')
    print()
    print(summary)

    # ------------------------------------Save torch step----------------------------------------

    latest_epoch += 1

    if epoch % 10 == 0 and epoch != 0:
        print("Saving Model" + str(epoch) + ".torch")  # Save model weight
        torch.save(model.state_dict(),
                   "./zoo/vit_player_classify/vit_classify_final_1122_" + str(latest_epoch) + ".torch")
