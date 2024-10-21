from PIL import Image
import numpy as np
import torch
from diffusers import DDIMScheduler
import cv2

import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))
from plugin_modules.DesignEdit.models.sdxl import sdxl
from plugin_modules.DesignEdit.models.inversion import Inversion
from plugin_modules.DesignEdit.models.layers import LayerFusion, Control, register_attention_control
from plugin_modules.DesignEdit.models import utils

# init hyper-paramter
# save_path = "./tmp_diffuser_plugin/design_edit"
save_path = "./tmp_example_img/watermark/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

model_type = "fp16"
pretrained_model_path = "./models/stable-diffusion-xl-base-1.0"
num_ddim_steps = 50
mask_time = [0, 40]
# mask_time = [0, 20]
# op_list = {}
# attend_scale = {}
# attend_scale = {}
device = "cuda"
mode = 'removal'
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False)
torch_dtype = torch.float16

# init model
ldm_model = sdxl.from_pretrained(pretrained_model_path, torch_dtype=torch_dtype, use_safetensors=True,
                                 variant=model_type, scheduler=scheduler)
ldm_model.to(device)
inversion = Inversion(ldm_model, num_ddim_steps)

# read image, original_image need numpy type
# image_name = "./tmp_example_img/ipadater_img/bear/babybear_fight.jpeg"
# mask_name = "./tmp_example_img/ipadater_img/bear/babybear_fight_mask.png"
image_name = "./tmp_example_img/watermark/watermark_panda.jpg"
mask_name = "./tmp_example_img/watermark/watermark_panda_mask.jpg"
original_image = cv2.imread(image_name)
mask = cv2.imread(mask_name, 0)
ori_shape = original_image.shape

prompt = ""

# zooming
attend_scale = 20
sample_ref_match = {0: 0, 1: 0}
op_list = None

# process image
image_gt = Image.fromarray(original_image).resize((1024, 1024))  # why 1024?
image_gt = np.stack([np.array(image_gt)])

remove_mask = utils.attend_mask(utils.convert_and_resize_mask(mask),  # 1024
                                attend_scale=attend_scale)  # numpy to tensor resize to 128x128
fg_mask_list = None
refine_mask = None

# 01-3: prepare: prompts, blend_time, refine_time
prompts = len(sample_ref_match) * [prompt]  # 2
blend_time = [0, 41]
# blend_time = [0, 20]
refine_time = [0, 25]

# 02: invert
_, x_t, x_stars, prompt_embeds, pooled_prompt_embeds = inversion.invert(image_gt, prompts, inv_batch_size=1)
# x_t : 1,4,128,128
# len(x_stars) 51, x_stars[0].shape,torch.Size([1, 4, 128, 128])
# prompt_embeds.shape 1, 77, 2048,prompt_embeds[0,:10,0],[-3.8926,  1.4365,  0.4290,  0.3218,  0.4766,  0.8999,  0.4238, -0.1031,
#          0.0088,  0.0685]
# pooled_prompt_embeds.shape,1, 1280,pooled_prompt_embeds[0,:10],[ 0.1418,  0.3132, -2.2715, -0.5449, -1.0391, -0.0795, -0.7046, -0.3328,
#         -1.0781,  0.1387]
# import pdb
# pdb.set_trace()
# 03: init layer_fusion and controller
lb = LayerFusion(remove_mask=remove_mask, fg_mask_list=fg_mask_list, blend_time=blend_time,
                 mode=mode, op_list=op_list)
controller = Control(layer_fusion=lb)
register_attention_control(model=ldm_model, controller=controller, mask_time=mask_time,
                           refine_time=refine_time)

# prompt = "two baby white bears are fighting"
# prompts = len(sample_ref_match) * [prompt]  # 2
# 04: generate images
images = ldm_model(controller=controller, prompt=prompts,
                   latents=x_t, x_stars=x_stars,
                   negative_prompt_embeds=prompt_embeds,
                   negative_pooled_prompt_embeds=pooled_prompt_embeds,
                   sample_ref_match=sample_ref_match)

gt_image = np.array(images[0])
inpainting_image = np.array(images[1])
resized_img = cv2.resize(inpainting_image, (ori_shape[1], ori_shape[0]))
cv2.imwrite(os.path.join(save_path, "watermark_panda_inpainting.jpg"), resized_img)
resized_gt = cv2.resize(gt_image, (ori_shape[1], ori_shape[0]))
cv2.imwrite(os.path.join(save_path, "watermark_panda_inpainting_gt.jpg"), resized_gt)
