import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from plugin_modules.instantstyle.ipadapter import IPAdapterXL
from plugin_modules.instantstyle.ipadapter.image_process import resize_img
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline, \
    StableDiffusionXLImg2ImgPipeline,StableDiffusionXLControlNetImg2ImgPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# initialization
base_model_path = "./models/stable-diffusion-xl-base-1.0"

controlnet_file_path = "/data/tc_guo/models/QRcodeCN"
controlnet = ControlNetModel.from_pretrained(controlnet_file_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    controlnet=controlnet,
).to(device)


pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

save_path = "./tmp_diffuser_plugin/instant_style_lighting/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

style_img_name = "./tmp_example_img/lighting/bg_indoor.png"
# prompt = "a cat is lying on the floor"  # a cat is lying on the floor
prompt = "an apple on the table"  # a cat is lying on the floor

pos_prompt = "(masterpiece:1.2),(best quality:1.2),"
neg_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, Extra limb"

# base image
base_image_name = "./tmp_example_img/lighting/merge_apple.jpg"
base_image = Image.open(base_image_name).convert("RGB")
base_image = resize_img(base_image)

style_img = Image.open(style_img_name)
# style_img = resize_img(style_img)
style_img = style_img.resize(base_image.size)
num_inference_steps = 20
neg_content_prompt = ""
neg_content_scale = 0.5
guidance_scale = 5.0
MAX_SEED = np.iinfo(np.int32).max
seed = random.randint(0, MAX_SEED)
scale = 0.5
strength = 0.7

# gen image

images = pipe(image=base_image, control_image=style_img,
              prompt=prompt + ',' + pos_prompt,
              negative_prompt=neg_prompt,
              guidance_scale=guidance_scale,
              num_images_per_prompt=4,
              num_inference_steps=num_inference_steps,
              seed=seed, strength=strength, ).images

for i, it in enumerate(images):
    save_name = os.path.join(save_path,
                             "QRlighting_img2img_s%0.2f_apple_prompt_%d.jpg" % (strength, i))
    it.save(save_name)
