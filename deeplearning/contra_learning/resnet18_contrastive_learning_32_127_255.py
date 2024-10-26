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


class ContrastivePlayerDataset(Dataset):
    def __init__(self, player_json_file, transform=None):
        with open(player_json_file, 'r') as rf:
            self.player_json = json.loads(rf.read())
        self.transform = transform

    def __len__(self):
        return len(self.player_json)

    def __getitem__(self, idx):
        pl_list = self.player_json[str(idx)]
        img = self.transform(Image.open(pl_list[0])).unsqueeze(dim=0)

        for pl_file in pl_list[1:]:
            img_t = self.transform(Image.open(pl_file)).unsqueeze(dim=0)
            img = torch.cat((img, img_t), dim=0)

        sample = {'images_tensor': img}

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
BATCH_SIZE = 1
SUB_BATCH_SIZE = 64
AVG_BATCH_SIZE = 50
train_json_file = "/home/wolf/datasets/reid/dataset/resnet/resnet18_final_dataset_32_127.json.20241025171219"

LOG_FILE = "./log/resnet/resnet18_cl_32_128.log." + get_nowtime_str()
SOFTMAX_LOG_FILE = "./log/resnet/resnet18_cl_32_128_softmax.log." + get_nowtime_str()
validation_rate = .1
shuffle_dataset = True
random_seed = 0

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((48, 96)),
    v2.ToTensor(),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ContrastivePlayerDataset(train_json_file, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device(
    'cpu')  # Set device GPU or CPU where the training will take place

# -------------------------Create model -------------------------------------------


model = torchvision.models.resnet18(pretrained=True)  # Load net
fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.LeakyReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 256),

)
model.fc = fc
model = model.to(device)
# read the resume weights
latest_model_file, latest_epoch = get_latest_model_file("./zoo/resnet/")
if latest_model_file:
    model.load_state_dict(torch.load(latest_model_file))

# -------------------------Create model -------------------------------------------

optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)  # Create adam optimizer
cos = nn.CosineSimilarity(dim=0, eps=1e-6)


def loss_func(y_batch):
    pair_loss = torch.exp(cos(y_batch[0, :], y_batch[1, :]))
    p1_loss = 0.
    p2_loss = 0.
    for i in range(2, y_batch.shape[0]):
        p1_loss += torch.exp(cos(y_batch[0, :], y_batch[i, :]))
        p2_loss += torch.exp(cos(y_batch[1, :], y_batch[i, :]))

    return -torch.log(pair_loss / (pair_loss + p1_loss + p2_loss))


def create_128x128_loss_mask():
    r1 = torch.cat((torch.tensor([0]), torch.ones(127)), dim=0)
    r2 = torch.cat((torch.tensor([0, 0]), torch.ones(126)), dim=0)
    r3 = torch.zeros(128)
    r3 = r3.repeat(126, 1)

    m = torch.cat((r1.unsqueeze(dim=0), r2.unsqueeze(dim=0), r3), dim=0)

    masked = m.bool()

    return masked


def cross_loss_func(y_batch, temperature=0.1):
    first_pair_loss_0 = cos(y_batch[0, :], y_batch[1, :]) / temperature
    loss_list = first_pair_loss_0.unsqueeze(dim=0)
    for i in range(2, y_batch.shape[0]):
        loss_list = torch.cat((loss_list, cos(y_batch[0, :], y_batch[i, :]).unsqueeze(dim=0) / temperature), dim=0)

    for i in range(2, y_batch.shape[0]):
        loss_list = torch.cat((loss_list, cos(y_batch[1, :], y_batch[i, :]).unsqueeze(dim=0) / temperature), dim=0)

    with open(SOFTMAX_LOG_FILE, 'a') as softmax_log:
        softmax_log.write(str(loss_list.detach().cpu().numpy()) + '\n')
    return criterion(loss_list.unsqueeze(dim=0), torch.tensor([0]).to(device))


def cross_loss_func_masked(features, masked, temperature=0.1):
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    logits = similarity_matrix[masked]
    logits = logits / temperature

    with open(SOFTMAX_LOG_FILE, 'a') as softmax_log:
        softmax_log.write(str(logits.detach().cpu().numpy()) + '\n')

    return criterion(logits.unsqueeze(dim=0), torch.tensor([0]).to(device))


# ----------------Train--------------------------------------------------------------------------

sub_train_idx = 0
notify_sub_batch_size = 5
email_notify_sub_batch_size = 500
len_datasize = len(train_dataset)
criterion = torch.nn.CrossEntropyLoss().to(device)
loss_masked = create_128x128_loss_mask().to(device)

for epoch in trange(TOTAL_EPOCHS + 1, desc="Training.."):  # Training loop
    train_loss = 0.0
    start = time.time()

    model.train()

    for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
    ):

        x_batch = batch['images_tensor'].squeeze()
        x_batch = x_batch.to(device)
        # total_num = x_batch.shape[0]
        # sub_batch_num = total_num // SUB_BATCH_SIZE
        # y_batch = model(x_batch[0:SUB_BATCH_SIZE, :])
        # batch_idx = SUB_BATCH_SIZE
        # while batch_idx < total_num:
        #     y_sub = model(x_batch[batch_idx:min(total_num, batch_idx + SUB_BATCH_SIZE), :])
        #     y_batch = torch.cat((y_batch, y_sub), dim=0)
        #     batch_idx += SUB_BATCH_SIZE
        y_batch = model(x_batch)

        # loss = cross_loss_func(y_batch)
        loss = cross_loss_func_masked(y_batch, loss_masked)

        train_loss = loss.detach().cpu().item()

        optimizer.zero_grad()
        loss.backward()

        sub_train_idx += 1

        if sub_train_idx % notify_sub_batch_size == 0:
            duration = time.time() - start
            summary = f"[{latest_epoch}/{TOTAL_EPOCHS}][{sub_train_idx}/{len_datasize}] train loss: {train_loss:.4f} , duration:{duration:.2f}"
            with open(LOG_FILE, 'a') as log:
                log.write(summary + '\n')
            print()
            print(summary)
            # if sub_train_idx % email_notify_sub_batch_size == 0:
            #     nowtime = datetime.datetime.now()
            #     try:
            #         send_mail("[" + nowtime.strftime("%Y%m%d") + "]-" + summary, 'ATT..............................')
            #     except Exception:
            #         print("Sending email failed!")
            #         print(traceback.format_exc())

            start = time.time()
    sub_train_idx = 0
    latest_epoch += 1

    print("Saving Model" + str(epoch) + ".torch")  # Save model weight
    torch.save(model.state_dict(), "./zoo/resnet/resnet18_cl_32_128_" + str(latest_epoch) + ".torch")

    # scheduler.step(val_loss)
