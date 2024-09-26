import json
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


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


json_file = "./player_annotation/player_annotated.json.20240924174013"

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((224, 224)),
    v2.ToTensor()
])

dataset = PlayerDataset(json_file, transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for sample in dataloader:
    print(sample["image_name"], sample["ground_truth"])
