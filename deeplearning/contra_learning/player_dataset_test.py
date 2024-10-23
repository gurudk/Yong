import os
import pathlib

import json
import random
import re
from pathlib import Path
import shutil

from torch.utils.data import Dataset, DataLoader
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

from itertools import combinations
from random import shuffle

np.random.seed(0)
torch.manual_seed(0)


class ContrastivePlayerDataset(Dataset):
    def __init__(self, player_json_file, transform=None):
        with open(player_json_file, 'r') as rf:
            self.player_json = json.loads(rf.read())
        self.transform = transform

    def __len__(self):
        return len(self.player_json)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        #
        # img_name = self.player_detections_items[idx][0]
        # image = Image.open(img_name)
        # angle = self.player_detections_items[idx][1]
        # gt_x = torch.cos(torch.tensor(angle))
        # gt_y = torch.sin(torch.tensor(angle))
        #
        # truth_tensor = torch.tensor([gt_x, gt_y]).squeeze()
        #

        pl_list = self.player_json[str(idx)]
        img = self.transform(Image.open(pl_list[0])).unsqueeze(dim=0)

        for pl_file in pl_list[1:]:
            img_t = self.transform(Image.open(pl_file)).unsqueeze(dim=0)
            img = torch.cat((img, img_t), dim=0)

        sample = {'images_tensor': img}

        return sample


json_file = "/home/wolf/datasets/reid/dataset/final_dataset.json.20241023152537"

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((48, 96)),
    v2.ToTensor(),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ContrastivePlayerDataset(json_file, transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for sample in dataloader:
    print(sample["images_tensor"].squeeze().shape)

    break
