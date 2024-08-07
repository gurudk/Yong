import torch
from vit_pytorch import ViT
import time

v = ViT(
    image_size=320,
    patch_size=20,
    num_classes=4,
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1
)

start = time.time()

for i in range(100):
    img = torch.randn(1, 3, 320, 180)

    preds = v(img)  # (1, 1000)

end = time.time()
print("Call duration:", str((end - start) / 100))
print(preds.shape)
