import os
import json
import time
import smtplib
import datetime
import traceback

from email.message import EmailMessage

from pathlib import Path

import numpy as np
import torch
import torchvision
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm, trange
from torchvision.transforms import v2

np.random.seed(0)
torch.manual_seed(0)

track_data_dir = "/home/wolf/datasets/reid/DFL/SR583_0"

# --------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')  # Set device GPU or CPU where the training will take place
model = torchvision.models.resnet18(pretrained=True)  # Load net
# fc = nn.Sequential(
#     nn.Linear(512, 512),
#     nn.LeakyReLU(),
#     nn.Dropout(p=0.5),
#     nn.Linear(512, 2),
#
# )

model.fc = nn.Identity()
model = model.to(device)
model.eval()

transform = v2.Compose([
    # you can add other transformations in this list
    v2.Resize((48, 96)),
    v2.ToTensor(),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

json_cluster = {}
track_numpy = np.empty((1, 512))
cluster_num = 0
for p in Path(track_data_dir).iterdir():
    if p.is_dir():
        p_dir_stem = p.stem
        track_tensor_list = []
        file_list = [f for f in p.iterdir() if f.is_file()]
        if len(file_list) >= 10:
            cluster_num += 1
            for fi in file_list:
                image = Image.open(str(fi))
                img = transform(image)
                img = img.to(device)
                feature = model(img.unsqueeze(dim=0))
                track_numpy = np.append(track_numpy, feature.cpu().detach().numpy(), axis=0)

            # json_cluster[p_dir_stem] = track_tensor_list

print(json_cluster)

ar_nan = np.where(np.isnan(track_numpy))
print("Nan", ar_nan)

ar_inf = np.where(np.isinf(track_numpy))
print("Inf", ar_inf)

# from tsnecuda import TSNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

matplotlib.use('QtAgg')

# tsne = TSNE(n_iter=1000, verbose=1, perplexity=10000, num_neighbors=64)
tsne = TSNE(n_components=2)
print(track_numpy.shape)
tsne_results = tsne.fit_transform(track_numpy)
y_train = range(track_numpy.shape[0])
df = {}
df['x'] = tsne_results[:, 0]
df['y'] = tsne_results[:, 1]

print(tsne_results.shape)
print(tsne_results)

sns.scatterplot(x='x', y='y', data=df)
plt.savefig("mygraph.png")
plt.show()

# # Create the figure
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1, title='TSNE')
#
# # Create the scatter
# ax.scatter(
#     x=tsne_results[:, 0],
#     y=tsne_results[:, 1],
#     c=y_train,
#     cmap=plt.cm.get_cmap('Paired'),
#     alpha=0.4,
#     s=0.5)
# plt.show()
