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
    StableDiffusionXLImg2ImgPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# initialization
base_model_path = "./models/stable-diffusion-xl-base-1.0"
# base_model_path = "./models/instantID_model/YamerMIX_v8"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"

# controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
# controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)

# load SDXL pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks"])

style_folder = "./tmp_example_img/instant_style/"
save_path = "./tmp_diffuser_plugin/instant_style_result/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

style_img_list = os.listdir(style_folder)
prompt = "shadow human"

pos_prompt = "(masterpiece:1.2),(best quality:1.2),"
neg_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"

# base image
base_image_name = "./tmp_example_img/fire_frame2.jpeg"
base_image = Image.open(base_image_name)
base_image = resize_img(base_image)

for it_style in style_img_list:
    if it_style.find("fire") == -1:
        continue
    style_img_name = os.path.join(style_folder, it_style)
    s_n, s_e = os.path.splitext(it_style)

    style_img = Image.open(style_img_name)
    style_img = resize_img(style_img)
    num_inference_steps = 20
    neg_content_prompt = ""  # "a lady portrait"  # a lady portrait
    neg_content_scale = 0.5
    guidance_scale = 5.0
    MAX_SEED = np.iinfo(np.int32).max
    seed = random.randint(0, MAX_SEED)
    scale = 1.0
    strength = 0.7

    # gen image
    if len(neg_content_prompt) > 0 and neg_content_scale != 0:
        images = ip_model.generate(pil_image=style_img,
                                   image=base_image,
                                   strength=0.7,
                                   prompt=prompt + ',' + pos_prompt,
                                   negative_prompt=neg_prompt,
                                   scale=scale,
                                   guidance_scale=guidance_scale,
                                   num_samples=4, num_inference_steps=num_inference_steps,
                                   neg_content_prompt=neg_content_prompt,
                                   neg_content_scale=neg_content_scale,
                                   seed=seed,
                                   )
    else:
        images = ip_model.generate(pil_image=style_img,
                                   image=base_image,
                                   strength=0.7,
                                   prompt=prompt + ',' + pos_prompt,
                                   negative_prompt=neg_prompt,
                                   scale=scale,
                                   guidance_scale=guidance_scale,
                                   num_samples=4,
                                   num_inference_steps=num_inference_steps,
                                   seed=seed,
                                   )
    for i, it in enumerate(images):
        save_name = os.path.join(save_path,
                                 "%s_%d.jpg" % (s_n, i))
        it.save(save_name)
