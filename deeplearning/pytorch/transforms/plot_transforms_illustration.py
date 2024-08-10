"""
==========================
Illustration of transforms
==========================

.. note::
    Try on `Colab <https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_transforms_illustrations.ipynb>`_
    or :ref:`go to the end <sphx_glr_download_auto_examples_transforms_plot_transforms_illustrations.py>` to download the full example code.

This example illustrates some of the various transforms available in :ref:`the
torchvision.transforms.v2 module <transforms>`.
"""
# %%

# sphinx_gallery_thumbnail_path = "../../gallery/assets/transforms_thumbnail.png"

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2

plt.rcParams["savefig.bbox"] = 'tight'

# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)

# If you're trying to run that on Colab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
from gallery.transforms.helpers import plot

orig_img = Image.open(Path('./gallery/assets') / 'astronaut.jpg')

# %%
# Geometric Transforms
# --------------------
# Geometric image transformation refers to the process of altering the geometric properties of an image,
# such as its shape, size, orientation, or position.
# It involves applying mathematical operations to the image pixels or coordinates to achieve the desired transformation.
#
# Pad
# ~~~
# The :class:`~torchvision.transforms.Pad` transform
# (see also :func:`~torchvision.transforms.functional.pad`)
# pads all image borders with some pixel values.

# padded_imgs = [v2.Pad(padding=padding)(orig_img) for padding in (3, 10, 30, 50)]
# plot([orig_img] + padded_imgs)

print(orig_img.width, orig_img.height)

# resized_imgs = [v2.Resize(size=size)(orig_img) for size in (30, 50, 100, 300, orig_img.size)]
# plot([orig_img] + resized_imgs)


# center_crops = [v2.CenterCrop(size=size)(orig_img) for size in (30, 50, 100, 256, 512, orig_img.size)]
# plot([orig_img] + center_crops)


# (top_left, top_right, bottom_left, bottom_right, center) = v2.FiveCrop(size=(256, 256))(orig_img)
# plot([orig_img] + [top_left, top_right, bottom_left, bottom_right, center])

# perspective_transformer = v2.RandomPerspective(distortion_scale=0.6, p=1.0)
# perspective_imgs = [perspective_transformer(orig_img) for _ in range(4)]
# plot([orig_img] + perspective_imgs)


# rotater = v2.RandomRotation(degrees=(0, 180))
# rotated_imgs = [rotater(orig_img) for _ in range(4)]
# plot([orig_img] + rotated_imgs)

# affine_transfomer = v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
# affine_imgs = [affine_transfomer(orig_img) for _ in range(4)]
# plot([orig_img] + affine_imgs)


# elastic_transformer = v2.ElasticTransform(alpha=250.0)
# transformed_imgs = [elastic_transformer(orig_img) for _ in range(2)]
# plot([orig_img] + transformed_imgs)


# cropper = v2.RandomCrop(size=(128, 128))
# crops = [cropper(orig_img) for _ in range(4)]
# plot([orig_img] + crops)


# resize_cropper = v2.RandomResizedCrop(size=(32, 32))
# resized_crops = [resize_cropper(orig_img) for _ in range(4)]
# plot([orig_img] + resized_crops)


# gray_img = v2.Grayscale()(orig_img)
# plot([orig_img, gray_img], cmap='gray')


# jitter = v2.ColorJitter(brightness=.5, hue=.3)
# jittered_imgs = [jitter(orig_img) for _ in range(4)]
# plot([orig_img] + jittered_imgs)


# inverter = v2.RandomInvert()
# invertered_imgs = [inverter(orig_img) for _ in range(4)]
# plot([orig_img] + invertered_imgs)


# posterizer = v2.RandomPosterize(bits=2)
# posterized_imgs = [posterizer(orig_img) for _ in range(4)]
# plot([orig_img] + posterized_imgs)


# solarizer = v2.RandomSolarize(threshold=12)
# solarized_imgs = [solarizer(orig_img) for _ in range(4)]
# plot([orig_img] + solarized_imgs)


# sharpness_adjuster = v2.RandomAdjustSharpness(sharpness_factor=20)
# sharpened_imgs = [sharpness_adjuster(orig_img) for _ in range(4)]
# plot([orig_img] + sharpened_imgs)


# autocontraster = v2.RandomAutocontrast()
# autocontrasted_imgs = [autocontraster(orig_img) for _ in range(10)]
# plot([orig_img] + autocontrasted_imgs)

# equalizer = v2.RandomEqualize()
# equalized_imgs = [equalizer(orig_img) for _ in range(4)]
# plot([orig_img] + equalized_imgs)

# jpeg = v2.JPEG((5, 50))
# jpeg_imgs = [jpeg(orig_img) for _ in range(4)]
# plot([orig_img] + jpeg_imgs)


# policies = [v2.AutoAugmentPolicy.CIFAR10, v2.AutoAugmentPolicy.IMAGENET, v2.AutoAugmentPolicy.SVHN]
# augmenters = [v2.AutoAugment(policy) for policy in policies]
# imgs = [
#     [augmenter(orig_img) for _ in range(4)]
#     for augmenter in augmenters
# ]
# row_title = [str(policy).split('.')[-1] for policy in policies]
# plot([[orig_img] + row for row in imgs], row_title=row_title)


# augmenter = v2.RandAugment()
# imgs = [augmenter(orig_img) for _ in range(4)]
# plot([orig_img] + imgs)


# augmenter = v2.TrivialAugmentWide()
# imgs = [augmenter(orig_img) for _ in range(4)]
# plot([orig_img] + imgs)


# augmenter = v2.AugMix()
# imgs = [augmenter(orig_img) for _ in range(4)]
# plot([orig_img] + imgs)


# hflipper = v2.RandomHorizontalFlip(p=0.5)
# transformed_imgs = [hflipper(orig_img) for _ in range(4)]
# plot([orig_img] + transformed_imgs)


# vflipper = v2.RandomVerticalFlip(p=0.5)
# transformed_imgs = [vflipper(orig_img) for _ in range(4)]
# plot([orig_img] + transformed_imgs)


applier = v2.RandomApply(transforms=[v2.RandomCrop(size=(64, 64))], p=0.5)
transformed_imgs = [applier(orig_img) for _ in range(4)]
plot([orig_img] + transformed_imgs)
