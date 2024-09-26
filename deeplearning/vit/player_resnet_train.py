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
            self.player_detections_items = [(key, value["target_angle"] * np.pi * 2) for key, value in
                                            self.annotations["player_detections"].items()]
        self.transform = transform

    def __len__(self):
        return len(self.player_detections_items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.player_detections_items[idx][0]
        image = Image.open(img_name)

        sample = {'image': image, 'ground_truth': self.player_detections_items[idx][1], 'image_name': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


def cos_similiar(y_pred, y):
    return 2 - 2 * torch.cos(y_pred - y)


LR = 1e-4
TOTAL_EPOCHS = 1000
BATCH_SIZE = 64
AVG_BATCH_SIZE = 50
json_file = "./player_annotation/player_annotated.json.20240924174013"

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((224, 224)),
    v2.ToTensor()
])

dataset = PlayerDataset(json_file, transform)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Set device GPU or CPU where the training will take place
model = torchvision.models.resnet34(pretrained=True)  # Load net
model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)  # Change final layer to predict one value
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)  # Create adam optimizer

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    'min',
    factor=0.5,
    patience=20,
    verbose=True,
    min_lr=1e-8,
)

# ----------------Train--------------------------------------------------------------------------
AverageLoss = np.ones([50])  # Save average loss for display
for epoch in trange(TOTAL_EPOCHS + 1, desc="Training.."):  # Training loop
    train_loss = 0.0
    start = time.time()

    for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
    ):
        x = batch['image']
        y = batch['ground_truth']

        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        loss = cos_similiar(y_pred.squeeze(), y).sum()

        train_loss += loss.detach().cpu().item() / len(train_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        AverageLoss[epoch % AVG_BATCH_SIZE] = loss.data.cpu().numpy() / BATCH_SIZE  # Save loss average
    duration = time.time() - start
    # print("\n")
    #
    # train_log = f"Epoch {epoch + 1}/{TOTAL_EPOCHS} loss: {train_loss:.5f} duration:{duration:.2f}"
    #
    # nowtime = datetime.datetime.now()
    # try:
    #     send_mail("[" + nowtime.strftime("%Y%m%d") + "]-" + train_log, 'ATT..............................')
    # except Exception:
    #     print("Sending email failed!")
    #     print(traceback.format_exc())
    #
    # print(f"Epoch {epoch + 1}/{TOTAL_EPOCHS} loss: {train_loss:.5f} duration:{duration:.2f}")
    #
    # if epoch % 100 == 0:
    #     torch.save(model, "./model/player_orientation_" + str(epoch) + ".pth")

    avgloss = AverageLoss.mean()
    summary = str(epoch) + "/" + str(TOTAL_EPOCHS) + " Loss:" + str(round(train_loss, 5)) + ' AverageLoss:' + str(
        round(avgloss, 5)) + " Current lr:" + str(scheduler.get_last_lr()) + " duration:" + str(round(duration, 2))
    # print(epoch, ") Loss=", train_loss, 'AverageLoss', avgloss, "Current lr rate:",
    #       scheduler.get_last_lr())  # Display loss
    print(summary)
    nowtime = datetime.datetime.now()
    try:
        send_mail("[" + nowtime.strftime("%Y%m%d") + "]-" + summary, 'ATT..............................')
    except Exception:
        print("Sending email failed!")
        print(traceback.format_exc())
    if epoch % 50 == 0:  # Save model
        print("Saving Model" + str(epoch) + ".torch", "last lr:", scheduler.get_last_lr())  # Save model weight
        torch.save(model.state_dict(), "./zoo/" + str(epoch) + ".torch")

    scheduler.step(avgloss)
