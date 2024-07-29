import torch
from diffusers import StableDiffusion3Pipeline

torch.cuda.empty_cache()

# from huggingface_hub import login
#
# login()

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()
image = pipe(
    "Ronaldo with long hair holding a sign on the football field that says Hello Zhang",
    negative_prompt="",
    num_inference_steps=100,
    guidance_scale=7.0,
).images[0]

image.save('m3_football player_ronaldo_longhair_100.jpg')
