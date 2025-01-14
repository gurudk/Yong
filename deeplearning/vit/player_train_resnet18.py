import os
import json
import time
import smtplib
import datetime
import traceback

from email.message import EmailMessage

import numpy as np
import torch
import torchvision
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from torchvision.transforms import v2

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


LR = 1e-5
TOTAL_EPOCHS = 1000
BATCH_SIZE = 64
AVG_BATCH_SIZE = 50
train_json_file = "./player_annotation/clean_body_orientation_loss25_mergeall_07.json.20240928093915"
val_json_file = "./player_annotation/clean_body_orientation_loss15_2nd_mergeall_val_07.json.20240928093915"
LOG_FILE = "./log/resnet18_train.log." + get_nowtime_str()

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((112, 112)),
    v2.ToTensor(),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = PlayerDataset(train_json_file, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = PlayerDataset(val_json_file, transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Set device GPU or CPU where the training will take place
model = torchvision.models.resnet18(pretrained=True)  # Load net
fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.LeakyReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 2),

)

model.fc = fc
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)  # Create adam optimizer

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    'min',
    factor=0.3,
    patience=50,
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

        train_loss += loss.detach().cpu().item() / len(train_dataset)
        train_theta_loss += train_th_loss.detach().cpu().item() / len(train_dataset)
        train_distance_loss += train_dis_loss.detach().cpu().item() / len(train_dataset)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:  # Save model
        print("Saving Model" + str(epoch) + ".torch")  # Save model weight
        torch.save(model.state_dict(), "./zoo/player_resnet18_clean15_" + str(epoch) + ".torch")

    # ------------------------------------valiadate step----------------------------------------
    print("----------------------------------validate step--------------------------")

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

            val_loss += loss.detach().cpu().item() / len(val_dataset)
            val_theta_loss += val_th_loss.detach().cpu().item() / len(val_dataset)
            val_distance_loss += val_dis_loss.detach().cpu().item() / len(val_dataset)

            idx += 1

    duration = time.time() - start

    summary = f"[{epoch}/{TOTAL_EPOCHS}] train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, duration:{duration:.2f}"
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
