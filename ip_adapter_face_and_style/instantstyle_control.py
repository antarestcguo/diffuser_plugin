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
    StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# initialization
base_model_path = "./models/stable-diffusion-xl-base-1.0"
# base_model_path = "./models/instantID_model/YamerMIX_v8"
base_model_file = "/data/tc_guo/models/sfmodels/animagine-xl-3.1.safetensors"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"

# load SDXL pipeline
controlnet_path = "./models/instantID_model/diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16).to(device)
# tmp_pipe = StableDiffusionXLPipeline.from_single_file(
#     base_model_file,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
# )
pipe = StableDiffusionXLControlNetPipeline.from_single_file(
    base_model_path,
    # unet=tmp_pipe.unet,
    # vae=tmp_pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    controlnet=controlnet,
)
# del tmp_pipe
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device,
                       target_blocks=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"])

save_path = "./tmp_diffuser_plugin/tmp_c_result/wolf"
if not os.path.exists(save_path):
    os.makedirs(save_path)

prompt = "a wolf walk on the loess plateau, cartoon style, best quality, masterpiece"

pos_prompt = ""
negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, low quality, long neck, frame, text, worst quality,watermark, deformed, ugly,  blur, out of focus,extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, "

# base image
# base_image_name = "./tmp_example_img/human_img/liuyifei.png"
# base_image = Image.open(base_image_name)
# base_image = resize_img(base_image)

ref_image_name = "./tmp_diffuser_plugin/tmp_c_result/wolf/2_cannycontrol_frame10.jpg"
canny_image_name = "./tmp_example_img/ipadater_img/wolf/canny_110.jpg"
# ref_image = Image.open(ref_image_name).convert("RGB")
canny_image = Image.open(canny_image_name).convert("RGB")

style_img = Image.open(ref_image_name)
style_img = resize_img(style_img)
num_inference_steps = 20
neg_content_prompt = ""  # "a lady portrait"  # a lady portrait
neg_content_scale = 0.5
guidance_scale = 5.0
MAX_SEED = np.iinfo(np.int32).max
seed = random.randint(0, MAX_SEED)
scale = 1.3
strength = 0.7

# gen image
if len(neg_content_prompt) > 0 and neg_content_scale != 0:
    images = ip_model.generate(pil_image=style_img,
                               # ontrol_image=canny_image,
                               image=canny_image,
                               prompt=prompt + ',' + pos_prompt,
                               negative_prompt=negative_prompt,
                               scale=scale,
                               guidance_scale=guidance_scale,
                               num_samples=4, num_inference_steps=num_inference_steps,
                               neg_content_prompt=neg_content_prompt,
                               neg_content_scale=neg_content_scale,
                               seed=seed,
                               )
else:
    images = ip_model.generate(pil_image=style_img,
                               # control_image=canny_image,
                               image=canny_image,
                               prompt=prompt + ',' + pos_prompt,
                               negative_prompt=negative_prompt,
                               scale=scale,
                               guidance_scale=guidance_scale,
                               num_samples=4,
                               num_inference_steps=num_inference_steps,
                               seed=seed,
                               )
for i, it in enumerate(images):
    save_name = os.path.join(save_path, "{}_cannycontrol_frame110.jpg".format(i))
    it.save(save_name)
