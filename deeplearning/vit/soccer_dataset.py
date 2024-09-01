import json
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


class SoccerDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.annotations = json.loads(f.read())
            self.dataarray = [(key, value) for key, value in self.annotations.items()]
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.dataarray[idx][0]
        image = Image.open(img_name)
        truth_tensor = torch.tensor(self.dataarray[idx][1]).reshape(-1, 4)
        sample = {'image': image, 'ground_truth': truth_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample


json_file = "./annotation/annotation_normalized.txt"

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((360, 640)),
    v2.ToTensor()
])

dataset = SoccerDataset(json_file, transform)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for sample in dataloader:
    print(sample['image'].shape)
    print(sample['ground_truth'])
    break
