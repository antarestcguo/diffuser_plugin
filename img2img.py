import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, \
    DDIMScheduler, AutoencoderKL
from PIL import Image

# from plugin_modules.ip_adapter import IPAdapterXL, IPAdapter

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image

base_model_path = "./models/stable-diffusion-xl-base-1.0"
# base_model_path = "./models/stable-diffusion-v1-5"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
# vae_model_path = "./models/IP-Adapter/stabilityai/sd-vae-ft-mse"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
# vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    base_model_path,
    # torch_dtype=torch.float16,
    # scheduler=noise_scheduler,
    # # vae=vae,
    # feature_extractor=None,
    # safety_checker=None
    add_watermarker=False,
).to(device)

# read image prompt
image = Image.open("./tmp_example_img/images/futurecity.png")
g_image = Image.open("./tmp_example_img/structure_controls/city_structure.jpeg")
save_path = "./tmp_ip_adpater_result"
if not os.path.exists(save_path):
    os.makedirs(save_path)
# load ip-adapter
# ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
images = pipe(num_samples=4, num_inference_steps=50,
                           # seed=42,
                           image=g_image, strength=0.6,prompt="a future city, with road, bridge, building, landscape scenes, Science and technology style,masterpiece, photographic, intricate detail, Ultra Detailed hyperrealistic real photo, 8k, high quality, hdr, studio lighting, professional, trending on artstation,",negative_prompt="obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, low quality, long neck, frame, text, big, 3d rendering, unreal"
                           ).images
for i, it in enumerate(images):
    save_name = os.path.join(save_path, "oriimg2img_loop1_%d.jpg" % i)
    it.save(save_name)
images = pipe(num_samples=4, num_inference_steps=50,
                           # seed=42,
                           image=images[0], strength=0.4,prompt="a future city, with road, bridge, building, landscape scenes, Science and technology style,masterpiece, photographic, intricate detail, Ultra Detailed hyperrealistic real photo, 8k, high quality, hdr, studio lighting, professional, trending on artstation,",negative_prompt="obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, low quality, long neck, frame, text, big, 3d rendering, unreal"
                           ).images


for i, it in enumerate(images):
    save_name = os.path.join(save_path, "oriimg2img_loop2_%d.jpg" % i)
    it.save(save_name)
