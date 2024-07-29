import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

from PIL import Image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
# pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

prompt = "A green Teddy bear sitting on a park bench on the left, high resolution"

# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
image = Image.open('./image_dog.png')
mask_image = Image.open('image_dog_mask.png')

init_image = image.convert("RGB")

images = pipe(prompt, image=init_image).images
images[0].save("refiner_green_teddy.jpg")
