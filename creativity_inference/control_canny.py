import sys
import os

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline, \
    StableDiffusionXLControlNetPipeline, ControlNetModel
# control inpainting, control img2img,
import torch
from PIL import Image
import cv2
import numpy as np

negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, low quality, long neck, frame, text, worst quality,watermark, deformed, ugly,  blur, out of focus,extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, "

device = "cuda"
save_path = "./tmp_diffuser_plugin/tmp_c_result/wolf"
if not os.path.exists(save_path):
    os.makedirs(save_path)
base_model_file = "/data/tc_guo/models/sfmodels/animagine-xl-3.1.safetensors"

controlnet_path = "./models/instantID_model/diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16).to(device)
pipe = StableDiffusionXLControlNetPipeline.from_single_file(
    base_model_file,
    torch_dtype=torch.float16,
    use_safetensors=True,
    controlnet=controlnet,
).to(device)
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

ref_image_name = "./tmp_example_img/ipadater_img/wolf/frame_10.jpg"
canny_image_name = "./tmp_example_img/ipadater_img/wolf/canny_10.jpg"
prompt = "a wolf walk on the loess plateau, cartoon style, best quality, masterpiece"
ref_image = Image.open(ref_image_name).convert("RGB")
canny_image = Image.open(canny_image_name).convert("RGB")

images = pipe(image=canny_image, num_images_per_prompt=4,
              num_inference_steps=20, prompt=prompt,
              negative_prompt=negative_prompt).images

# save
for i, it in enumerate(images):
    save_name = os.path.join(save_path, "{}_cannycontrol_frame10.jpg".format(i))
    it.save(save_name)
