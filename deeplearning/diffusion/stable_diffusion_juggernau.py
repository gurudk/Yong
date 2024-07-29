import torch
from diffusers import DiffusionPipeline
from PIL import Image

pipeline = DiffusionPipeline.from_pretrained("RunDiffusion/Juggernaut-XL-v9")

pipeline.enable_model_cpu_offload()

# prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
prompt = "a living room with windows and sea in the background, sunset, high resolution"

# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
image = Image.open('6M0A3501_1920x1280.jpg').convert("RGB")
mask_image = Image.open('6M0A3501_inverted_mask.png')

images = pipeline(prompt=prompt, image=image, mask_image=mask_image).images
print(len(images))
images[0].save("./sofa_juggernau_01.png")
