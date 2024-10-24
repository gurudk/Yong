import einops
import numpy as np

images = [np.random.randn(360, 640, 3) for _ in range(1)]

print(einops.rearrange(images, 'b h w c -> b h w c').shape)

print(einops.rearrange(images, 'b (h1 h) (w1 w) c -> b (h1 w1) (h w c)', h1=20, w1=20).shape)

print(einops.rearrange(images, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=9, p2=16).shape)
