from PIL import Image
import numpy as np
import torch
from diffusers import DDIMScheduler
import cv2

import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))
from plugin_modules.DesignEdit.models.inversion import Inversion
from plugin_modules.DesignEdit.models.img2imgsdxl import img2imgsdxl
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline, \
    StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline, \
    StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# initialization
base_model_path = "./models/stable-diffusion-xl-base-1.0"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False)
# pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     scheduler=scheduler,
# ).to(device)
# model_type = "fp16"
# pipe = sdxl.from_pretrained(base_model_path, torch_dtype=torch.float16, use_safetensors=True,
#                                  variant=model_type, scheduler=scheduler)

# pipe.enable_vae_slicing()
# pipe.enable_xformers_memory_efficient_attention()

num_ddim_steps = 50

# init model
torch_dtype = torch.float16
model_type = "fp16"
ldm_model = img2imgsdxl.from_pretrained(base_model_path, torch_dtype=torch_dtype, use_safetensors=True,
                                        variant=model_type, scheduler=scheduler)
ldm_model.to(device)
inversion = Inversion(ldm_model, num_ddim_steps)

save_path = "./tmp_diffuser_plugin/design_edit/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

inversion = Inversion(ldm_model, num_ddim_steps)

prompt = "a cat is lying on the floor"  # a cat is lying on the floor
image_name = "./tmp_example_img/lighting/merge.png"
original_image = Image.open(image_name).convert("RGB")

# process image
image_gt = original_image.resize((1024, 1024))  # why 1024?
image_gt = np.stack([np.array(image_gt)])
prompts = prompt  # 2
# 02: invert
_, x_t, x_stars, prompt_embeds, pooled_prompt_embeds = inversion.invert(image_gt, "", inv_batch_size=1)
# x_stars[-1] = x_t

# save results
# for i, it in enumerate(x_stars):
#     save_name = os.path.join(save_path,"x_stars_%d.jpg"%i)
#     save_img = inversion.latent2image(it.to(torch.float32))[0][:,:,::-1]
#     cv2.imwrite(save_name,save_img)
import pdb

pdb.set_trace()

images = ldm_model(prompt=prompt,
                   latents=x_stars[-31], x_stars=x_stars,
                   negative_prompt_embeds=prompt_embeds,
                   negative_pooled_prompt_embeds=pooled_prompt_embeds,
                   strength=0.4
                   )

for i, it in enumerate(images):
    save_name = os.path.join(save_path,
                             "inversion_img2img_cat_prompt_%d_tmp04.jpg" % (i))
    it.save(save_name)
