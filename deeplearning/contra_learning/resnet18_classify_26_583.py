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
            self.player_dict = self.player_dataset["player_dicts"]
            self.player_items = list(self.player_json.items())
        self.transform = transform

    def __len__(self):
        return len(self.player_json)

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
full_dataset_file = "/home/wolf/datasets/reid/dataset/classify/player_classify_span3.json.20241107125947"

LOG_FILE = "./log/classify/resnet18_classify_26_583.log." + get_nowtime_str()
SOFTMAX_LOG_FILE = "./log/classify/resnet18_classify_26_583_softmax.log." + get_nowtime_str()
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


model = torchvision.models.resnet18(pretrained=True)  # Load net
fc = nn.Sequential(
    nn.Linear(512, 100)

)
model.fc = fc
model = model.to(device)
# read the resume weights
latest_model_file, latest_epoch = get_latest_model_file("./zoo/classify/")
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

        y_pre = model(x_batch)

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
        idx = 0
        for batch in val_loader:
            x_batch = batch['image_tensor'].squeeze()
            x_truth = batch['ground_truth'].squeeze()
            x_batch = x_batch.to(device)
            x_truth = x_truth.to(device)

            y_pre = model(x_batch)

            loss = criterion(y_pre, x_truth)

            val_loss += loss.detach().cpu().item() / len(val_indices)

            idx += 1

    scheduler.step(val_loss)
    # ------------------------------------print summary step----------------------------------------
    duration = time.time() - start
    summary = f"[{latest_epoch}/{TOTAL_EPOCHS}] train loss: {train_loss:.4f} , val loss: {val_loss:.4f} , duration:{duration:.2f}"
    with open(LOG_FILE, 'a') as log:
        log.write(summary + '\n')
    print()
    print(summary)

    # ------------------------------------Save torch step----------------------------------------

    latest_epoch += 1

    if epoch % 5 == 0:
        print("Saving Model" + str(epoch) + ".torch")  # Save model weight
        torch.save(model.state_dict(), "./zoo/classify/resnet18_classify_26_583_" + str(latest_epoch) + ".torch")
