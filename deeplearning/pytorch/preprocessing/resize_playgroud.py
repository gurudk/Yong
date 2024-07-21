from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

plt.rcParams["savefig.bbox"] = 'tight'

# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)

# If you're trying to run that on Colab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/

orig_img = Image.open(Path('./96.png'))
resize_func = T.Resize((360, 640))

resize_img = resize_func(orig_img)

resize_img.save('resized_360x640.png')
