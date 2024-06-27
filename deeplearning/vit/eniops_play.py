import numpy as np
import einops as E

images = [np.random.randn(30, 40, 3) for _ in range(32)]
print(images[0].shape)

print(E.rearrange(images, 'b h w c -> b h w c').shape)

print(E.rearrange(images, 'b h w c -> b (h w) c').shape)

print(E.rearrange(images, 'b h w c -> b w h c').shape)

print(E.rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape)

print(E.rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape)
