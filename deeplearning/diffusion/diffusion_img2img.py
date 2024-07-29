import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

img = Image.open("./6M0A3501.jpg")
img = img.resize((960, 640))
init_image = img.convert("RGB")
prompt = "a photo of a poster with couch, window an sea, very relax style"
images = pipe(prompt, image=init_image).images
# print(len(images))
images[0].save("horse3.jpg")
