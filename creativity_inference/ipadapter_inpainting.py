import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))
from plugin_modules.ip_adapter.ip_adapter import IPAdapterXL, IPAdapterPlusXL
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline

import torch
from PIL import Image

negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, low quality, long neck, frame, text, worst quality,watermark, deformed, ugly,  blur, out of focus,extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, "

device = "cuda"
save_path = "./tmp_diffuser_plugin/tmp_c_result"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# base model
base_model_file = "/data/tc_guo/models/sfmodels/samaritan3dCartoon_v40SDXL.safetensors"
base_model_file = "/data/tc_guo/models/sfmodels/dreamshaperXL10_alpha2Xl10.safetensors"
base_model_path = "/data/tc_guo/models/stable-diffusion-xl-1.0-inpainting-0.1"
# base_model_file = "/data/tc_guo/models/sfmodels/Juggernaut.safetensors"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"

lora_path = "/data/tc_guo/models/sfmodels/cartoon_stickers_xl_v1.safetensors"
b_lora = True
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
# pipe = StableDiffusionXLInpaintPipeline.from_single_file(
#     base_model_file,
#     torch_dtype=torch.float16,
#     add_watermarker=False,
# )
if b_lora:
    pipe.load_lora_weights(lora_path)


a = 0
# xlplus
image_encoder_refine_path = "./models/IP-Adapter/models/image_encoder"
ip_ckpt_refine = "./models/IP-Adapter/sdxl_model/ip-adapter-plus_sdxl_vit-h.safetensors"
pipe = StableDiffusionXLInpaintPipeline.from_single_file(
    base_model_file,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

img_path = ''
mask_path = ''

# json or image
tmp_n, tmp_e = os.path.splitext(mask_path)
if tmp_e == ".json":
    a = 0
else:
    mask = Image.open(mask_path)
