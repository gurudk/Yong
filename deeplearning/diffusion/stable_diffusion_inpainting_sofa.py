import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

from PIL import Image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
# pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

# prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
prompt = "a living room with windows and sea in the background, sunset, high resolution"

# image and mask_image should be PIL images.
# The mask structure is white for inpainting and black for keeping as is
image = Image.open('6M0A3501_1920x1280.jpg').convert("RGB")
mask_image = Image.open('6M0A3501_inverted_mask.png')

images = pipe(prompt=prompt, image=image, mask_image=mask_image).images
print(len(images))
images[0].save("./sofa_03.png")
