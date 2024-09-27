import json

import numpy as np
import torch
import torchvision
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

np.random.seed(0)
torch.manual_seed(0)


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


BATCH_SIZE = 1
AVG_BATCH_SIZE = 50
json_file = "./player_annotation/split_3000_player.json.20240927115232"

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((224, 224)),
    v2.ToTensor()
])

val_dataset = PlayerDataset(json_file, transform)

# sub_val_dataset = torch.utils.data.Subset(val_dataset, range(0, len(val_dataset), 10))

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------Load and set net and optimizer-------------------------------------
# --------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Set device GPU or CPU where the training will take place
model = torchvision.models.resnet18(pretrained=True)  # Load net
# model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)  # Change final layer to predict one value
fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 1),
    # nn.Hardtanh(min_val=0, max_val=np.pi * 2)
    nn.Sigmoid()
)
model.fc = fc

model = model.to(device)
model.load_state_dict(torch.load("./zoo/150.torch"))

# ----------------Train--------------------------------------------------------------------------
AverageLoss = np.ones([AVG_BATCH_SIZE])  # Save average loss for display
val_loss = 0
idx = 0
for batch in val_loader:
    x = batch['image']
    y = batch['ground_truth']

    x, y = x.to(device), y.to(device)
    y_pred = model(x)

    loss = cos_similiar(y_pred.squeeze(), y).sum()
    val_loss += loss.detach().cpu().item() / len(val_dataset)
    AverageLoss[idx % AVG_BATCH_SIZE] = loss.data.cpu().numpy() / BATCH_SIZE

    idx += 1
    print("Last ", AVG_BATCH_SIZE, " loss:", AverageLoss.mean())

print("val loss:", val_loss)
