import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

# pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("/home/wolf/models/sd_xl_refiner_1.0.safetensors", from_pt=True)
pipe = pipe.to("cuda")
# url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
#
# init_image = load_image(url).convert("RGB")
init_image = Image.open("6M0A3501.jpg")
init_image.resize((960, 640))
init_image.save("6M0A3501_960x640.jpg")
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, image=init_image).images
