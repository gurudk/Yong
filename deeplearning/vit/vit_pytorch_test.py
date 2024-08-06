import torch
from vit_pytorch import ViT
import time

v = ViT(
    image_size=256,
    patch_size=16,
    num_classes=4,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)

start = time.time()

for i in range(100):
    img = torch.randn(1, 3, 256, 144)

    preds = v(img)  # (1, 1000)

end = time.time()
print("Call duration:", str((end - start) / 100))
print(preds.shape)
