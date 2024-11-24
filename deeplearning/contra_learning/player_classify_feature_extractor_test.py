import itertools

import torch

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
from pathlib import Path
from random import shuffle

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


def get_model_from_torchfile(torch_file, device, out_dim):
    model = PlayerClassifyViT(
        image_size=(48, 96),
        patch_size=(3, 6),
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
        out_dim=out_dim
    )

    model.load_state_dict(torch.load(torch_file))
    model = model.to(device)
    return model


def get_features(img_name, model, device):
    img = Image.open(img_name)
    img_tensor = transform(img).unsqueeze(dim=0)
    img_tensor = img_tensor.to(device)

    _, y_features = model(img_tensor)
    return y_features


def get_sim_by_category(src_image_name, dest_category_dir, model, device):
    f_src = get_features(src_image_name, model, device)
    sim_list = list()
    for p in Path(dest_category_dir).iterdir():
        if p.is_file():
            f_dest = get_features(str(p), model, device)
            simi = cosi(f_src, f_dest).detach().cpu().item()
            sim_list.append(simi)

    return np.array(sim_list).mean()


def get_sim_by_target_list(src_image_name, target_list, model, device):
    f_src = get_features(src_image_name, model, device)
    sim_list = list()
    for target_file in target_list:
        f_dest = get_features(target_file, model, device)
        simi = cosi(f_src, f_dest).detach().cpu().item()
        sim_list.append(simi)

    return np.array(sim_list).mean()


def get_shuffle_images_by_num(input_dir, file_num=30):
    file_list = list()
    for p in Path(input_dir).iterdir():
        if p.is_file():
            file_list.append(str(p))
    shuffle(file_list)
    return file_list[:file_num]


torch_file = "./zoo/vit_player_classify/vit_classify_final_1122_301.torch"
full_dataset_file = "/home/wolf/datasets/reid/dataset/classify/train_classify_minnum20_span3.json.20241122105138"

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((48, 96)),
    v2.ToTensor(),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_dataset = PlayerClassifyDataset(full_dataset_file, transform)
# --------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device(
    'cpu')  # Set device GPU or CPU where the training will take place

model = get_model_from_torchfile(torch_file, device, full_dataset.get_class_num())

# print(model)

test_img_name = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241124165612/SR26_team_b_b1/SR26_1_95_frame_404_670.png"
test_img_name1 = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241124165612/SR26_team_b_b1/SR26_1_125_frame_1126_626.png"
test_img_name2 = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241124165612/SR200A_team_w_w13/SR200A_0_48_frame_663_722.png"

f = get_features(test_img_name, model, device)
f1 = get_features(test_img_name1, model, device)
f2 = get_features(test_img_name2, model, device)
cosi = torch.nn.CosineSimilarity(dim=1)
f_f1 = cosi(f, f1)
f_f2 = cosi(f, f2)
print(f_f1, f_f2)

cat1_dir = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241124165612/SR475_team_b_b9/"
cat1_dir1 = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241124165612/SR583_team_b_b6/"
cat1_dir1_sample1 = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241124165612/SR583_team_b_b6/SR583_1_118_frame_873_633.png"
cat1_dir2 = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241124165612/SR694_team_p_p27/"
cat1_dir_list = list()
for p in Path(cat1_dir).iterdir():
    if p.is_file():
        cat1_dir_list.append(p)

print(cat1_dir_list)
idxs = range(len(cat1_dir_list))

cat1_dir_pairs = list(itertools.combinations(idxs, 2))
cat1_res_list = list()
for pr in cat1_dir_pairs[:100]:
    this_filename = str(cat1_dir_list[pr[0]])
    other_filename = str(cat1_dir_list[pr[1]])
    this_features = get_features(this_filename, model, device)
    other_features = get_features(other_filename, model, device)
    simi = cosi(this_features, other_features)
    simi = simi.detach().cpu().item()
    cat1_res_list.append(simi)
    # print(simi)

np_cat1_res_list = np.array(cat1_res_list)
print(np_cat1_res_list.mean())

zz_res_mean = get_sim_by_category(cat1_dir1_sample1, cat1_dir, model, device)
print("sample1 with dir simi mean:", zz_res_mean)
print("sample1 with dir2 simi mean:", get_sim_by_category(cat1_dir1_sample1, cat1_dir2, model, device))
print("sample1 with dir1 simi mean:", get_sim_by_category(cat1_dir1_sample1, cat1_dir1, model, device))

# base_dir = "/home/wolf/datasets/reid/dataset/classsify_dump_dir_20241124165612/"
# for p in Path(base_dir).iterdir():
#     if p.is_dir():
#         sub_dir = str(p).split("/")[-1]
#         print(sub_dir, get_sim_by_category(cat1_dir1_sample1, str(p), model, device))

test_zz_file = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/SR21119_0/SR21119_0_12_w36_player/SR21119_0_12_frame_43_723.png"
test_base_dir = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/SR21119_0/"
test_zz_file1 = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/SR21119_0/SR21119_0_76_w5_player/SR21119_0_76_frame_804_739.png"
test_zz_file2 = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/SR21119_0/SR21119_0_82_g8_player/SR21119_0_82_frame_891_787.png"
test_zz_file3 = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/SR21119_0/SR21119_0_86_g6_player/SR21119_0_86_frame_953_776.png"

test_base_dir1 = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/SR21119_1/"
test_base_dir2 = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/SR21119_2/"
# res_dict = {}
# for p in Path(test_base_dir1).iterdir():
#     if p.is_dir():
#         sub_dir = str(p).split("/")[-1]
#         res_dict[sub_dir] = get_sim_by_category(test_zz_file3, str(p), model, device)
#
# res_items = list(res_dict.items())
# res_items.sort(key=lambda itm: itm[1])
# for item in res_items:
#     print(item)

test_player_w36_dir = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/merge/w36/"
test_player_w77_dir = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/merge/w77/"
test_player_g11_dir = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/merge/g11/"
test_player_g8_dir = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/SR21119_0/SR21119_0_82_g8_player"
test_player_g6_dir = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/SR21119_0/SR21119_0_2_g6_player"
test_player_g37_dir = "/home/wolf/datasets/reid/DFL/dest_manual/SR21119/merge/g37"
src_g6_files = get_shuffle_images_by_num(test_player_g6_dir, file_num=15)

src_g37_files = get_shuffle_images_by_num(test_player_g37_dir, file_num=30)
src_g8_files = get_shuffle_images_by_num(test_player_g8_dir, file_num=15)
src_g11_files = get_shuffle_images_by_num(test_player_g11_dir, file_num=30)
src_w36_files = get_shuffle_images_by_num(test_player_w36_dir, file_num=30)
src_w77_files = get_shuffle_images_by_num(test_player_w77_dir, file_num=30)
res_dict = {}
for p in Path(test_base_dir2).iterdir():
    if p.is_dir():
        target_list = get_shuffle_images_by_num(str(p), file_num=30)
        sub_simi_list = list()
        sub_dir = str(p).split("/")[-1]
        for src_file in src_g37_files:
            sub_simi_list.append(get_sim_by_target_list(src_file, target_list, model, device))
        res_dict[sub_dir] = np.array(sub_simi_list).mean()
        print(sub_dir, " calculate completed~")

print("=======================================summary==============================")
res_items = list(res_dict.items())
res_items.sort(key=lambda itm: itm[1])
for item in res_items:
    print(item)
