import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from PIL import Image
from torchvision.transforms import v2
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision.ops.boxes import box_area
from torchvision.ops import generalized_box_iou, generalized_box_iou_loss

np.random.seed(0)
torch.manual_seed(0)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def loss_func(src_box: object, target_box: object) -> object:
    return 1 - generalized_box_iou(src_box, target_box)


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
        # truth_tensor = torch.tensor(self.dataarray[idx][1]).reshape(-1, 4)
        truth_tensor = torch.tensor(self.dataarray[idx][1])
        sample = {'image': image, 'ground_truth': truth_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class SoccerViT(nn.Module):
    def __init__(self, chw, n_patches=40, n_blocks=6, hidden_d=256, n_heads=8, out_d=4):
        # Super constructor
        super(SoccerViT, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert (
                chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
                chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        patch_height = self.chw[1] // n_patches
        patch_width = self.chw[2] // n_patches
        patch_dim = self.chw[0] * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (p1 h) (p2 w) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_d),
            nn.LayerNorm(hidden_d),
        )

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches ** 2 + 1, hidden_d),
            persistent=False,
        )

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # 5) Classification MLPk
        # self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1))

        self.mlp = MLP(self.hidden_d, self.hidden_d, out_d, 3)

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        # patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        # tokens = self.linear_mapper(patches)
        tokens = self.to_patch_embedding(images)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]
        outputs_coord = self.mlp(out).sigmoid()
        return outputs_coord


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


def main():
    # Loading data
    json_file = "./annotation/annotation_normalized.txt"

    transform = v2.Compose([
        # you can add other transformations in this list
        v2.Resize((360, 640)),
        v2.ToTensor()
    ])
    train_set = SoccerDataset(json_file, transform)
    # test_set = MNIST(
    #     root="./../datasets", train=False, download=True, transform=transform
    # )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=16)
    # test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )
    # model = SoccerViT(
    #     (1, 28, 28), n_patches=7, n_blocks=2, hidden_d=16, n_heads=2, out_d=10
    # ).to(device)

    model = SoccerViT(
        (3, 360, 640), n_patches=40, n_blocks=6, hidden_d=256, n_heads=8, out_d=4
    ).to(device)

    N_EPOCHS = 1000
    LR = 0.001

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        start = time.time()

        for batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):
            x = batch['image']
            y = batch['ground_truth']

            x, y = x.to(device), y.to(device)
            pre_cord = model(x)
            y_hat = box_cxcywh_to_xyxy(pre_cord)
            loss = generalized_box_iou_loss(y_hat, y)
            loss = loss.sum()

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        duration = time.time() - start
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f} duration:{duration:.2f}")

    # Test loop
    # with torch.no_grad():
    #     correct, total = 0, 0
    #     test_loss = 0.0
    #     for batch in tqdm(test_loader, desc="Testing"):
    #         x, y = batch
    #         x, y = x.to(device), y.to(device)
    #         y_hat = model(x)
    #         loss = criterion(y_hat, y)
    #         test_loss += loss.detach().cpu().item() / len(test_loader)
    #
    #         correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
    #         total += len(x)
    #     print(f"Test loss: {test_loss:.2f}")
    #     print(f"Test accuracy: {correct / total * 100:.2f}%")
    torch.save(model, 'soccer_vit_1000.pth')


if __name__ == "__main__":
    main()
