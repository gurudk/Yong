from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
# prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
prompt = "A green Teddy bear sitting on a park bench on the left, high resolution"

# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
image = Image.open('./image_dog.png')
mask_image = Image.open('image_dog_mask.png')
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./green_teddy_on_park_bench.png")
