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
save_path = "./tmp_diffuser_plugin/design_edit"
if not os.path.exists(save_path):
    os.makedirs(save_path)

model_type = "fp16"
pretrained_model_path = "./models/stable-diffusion-xl-base-1.0"
num_ddim_steps = 50
mask_time = [0, 40]
# op_list = {}
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
original_image = cv2.imread("./tmp_example_img/human_img/liuyifei.png")
height_scale = 1.1
width_scale = 1.3
prompt = ""

# zooming
op_list = {0: ['zooming', [height_scale, width_scale]]}
ori_shape = original_image.shape
attend_scale = 30
sample_ref_match = {0: 0, 1: 0}


# 01-2: prepare: image_gt, remove_mask, fg_mask_list, refine_mask
img_height, img_width, _ = original_image.shape
new_height = int(img_height * height_scale)
new_width = int(img_width * width_scale)
mask = 255 * np.ones((new_height, new_width), dtype=np.uint8)
resize_img = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
# resize_img = np.zeros([new_height,new_width,3],dtype=np.uint8) #  no fill
x_offset = (new_width - img_width) // 2
y_offset = (new_height - img_height) // 2
resize_img[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = original_image
mask[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = 0

# img_new, mask = utils.zooming(original_image, [height_scale, width_scale]) # return nparray
img_new_copy = resize_img.copy()
mask_copy = mask.copy()
image_gt = Image.fromarray(resize_img).resize((1024, 1024))  # why 1024?
image_gt = np.stack([np.array(image_gt)])

remove_mask = utils.attend_mask(utils.convert_and_resize_mask(mask),  # 1024
                                attend_scale=attend_scale)  # numpy to tensor resize to 128x128
fg_mask_list = None
refine_mask = None

# 01-3: prepare: prompts, blend_time, refine_time
prompts = len(sample_ref_match) * [prompt]  # 2
blend_time = [0, 41]
refine_time = [0, 25]

# 02: invert
_, x_t, x_stars, prompt_embeds, pooled_prompt_embeds = inversion.invert(image_gt, prompts, inv_batch_size=1)

# 03: init layer_fusion and controller
lb = LayerFusion(remove_mask=remove_mask, fg_mask_list=fg_mask_list, blend_time=blend_time,
                 mode=mode, op_list=op_list)
controller = Control(layer_fusion=lb)
register_attention_control(model=ldm_model, controller=controller, mask_time=mask_time,
                           refine_time=refine_time)

# 04: generate images
images = ldm_model(controller=controller, prompt=prompts,
                   latents=x_t, x_stars=x_stars,
                   negative_prompt_embeds=prompt_embeds,
                   negative_pooled_prompt_embeds=pooled_prompt_embeds,
                   sample_ref_match=sample_ref_match)

gt_image = np.array(images[0])
outpainting_image = np.array(images[1])
resized_img = cv2.resize(outpainting_image, (new_width, new_height))
cv2.imwrite(os.path.join(save_path, "outpainting.jpg"), resized_img)
resized_gt = cv2.resize(gt_image, (new_width, new_height))
cv2.imwrite(os.path.join(save_path, "outpainting_gt.jpg"), resized_gt)
