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

# fg image
fg_prompt = "a cat"  # a cat is lying on the floor
fg_image_name = "./tmp_example_img/lighting/fg_cat_small.png"
fg_image = Image.open(fg_image_name).convert("RGB")

# process image
fg_image = fg_image.resize((1024, 1024))  # why 1024?
fg_image = np.stack([np.array(fg_image)])

# bg image
bg_prompt = "indoor floor"  # a cat is lying on the floor
bg_image_name = "./tmp_example_img/lighting/bg_indoor.png"
bg_image = Image.open(bg_image_name).convert("RGB")

# process image
bg_image = bg_image.resize((1024, 1024))  # why 1024?
bg_image = np.stack([np.array(bg_image)])

# direct merge image
prompt = "a cat is lying on the floor"  # a cat is lying on the floor
image_name = "./tmp_example_img/lighting/merge.png"
image = Image.open(fg_image_name).convert("RGB")

# process image
image = image.resize((1024, 1024))  # why 1024?
image = np.stack([np.array(image)])

# process mask
mask_image_name = "./tmp_example_img/lighting/mask_fg_cat_small.jpg"
mask_image = Image.open(fg_image_name)
mask_image = mask_image.resize((1024, 1024))
alpha = np.array(mask_image.split()[-1])

# 02: invert fg
_, x_t_fg, x_stars_fg, prompt_embeds_fg, pooled_prompt_embeds_fg = inversion.invert(fg_image, "", inv_batch_size=1)
# x_stars[-1] = x_t

# invert bg
_, x_t_bg, x_stars_bg, prompt_embeds_bg, pooled_prompt_embeds_bg = inversion.invert(bg_image, "", inv_batch_size=1)

# invert direct merge
_, x_t_merge, x_stars_merge, prompt_embeds_merge, pooled_prompt_embeds_merge = inversion.invert(image, prompt, inv_batch_size=1)

# merge fg and bg
alpha = cv2.resize(alpha, x_t_fg.shape[-2:], interpolation=cv2.INTER_NEAREST)
alpha = torch.from_numpy(alpha).to(device).to(x_t_fg.dtype) / 255
x_t = alpha * x_t_fg + (1 - alpha) * x_t_bg
x_stars = []
for it_fg, it_bg in zip(x_stars_fg, x_stars_bg):
    x_stars.append(alpha * it_fg + (1 - alpha) * it_bg)

prompt = "a cat is lying on the floor"

images = ldm_model(prompt=prompt,
                   latents=x_stars[-31], x_stars=x_stars,
                   negative_prompt_embeds=prompt_embeds_merge,
                   negative_pooled_prompt_embeds=pooled_prompt_embeds_merge,
                   strength=0.4
                   )

for i, it in enumerate(images):
    save_name = os.path.join(save_path,
                             "inversion_split_img2img_cat_prompt_%d_tmp04.jpg" % (i))
    it.save(save_name)
