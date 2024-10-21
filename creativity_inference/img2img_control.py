from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline,ControlNet,StableDiffusionXLControlNetPipeline,StableDiffusionXLControlNetImg2ImgPipeline,ControlNetModel
import torch
from PIL import Image
import argparse
from random import choice, shuffle

b_file = False
b_lora = False

base_model_path = "./models/stable-diffusion-xl-base-1.0"
base_model_file = "/data/tc_guo/models/sfmodels/samaritan3dCartoon_v40SDXL.safetensors"
controlnet_file_path = "/data/tc_guo/models/QRcodeCN"
lora_path = ""

controlnet = ControlNetModel.from_pretrained(controlnet_file_path, torch_dtype=torch.float16)

img_name = ""
save_path = ""

if b_file:
    base_pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        base_model_file,
        torch_dtype=torch.float16,
        add_watermarker=False,
        controlnet=controlnet,
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
        controlnet=controlnet,
        vae=base_pipe.vae,
    )
    del base_pipe
else:
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
        controlnet=controlnet,
    )

if b_lora:
    pipe.load_lora_weights(lora_path)