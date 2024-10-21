import torch
import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from plugin_modules.selfattn.pipeline_xl import StableDiffusionXLPipeline
from plugin_modules.DesignEdit.models.inversion import Inversion

import PIL.Image as Image
import numpy as np

base_model_path = "./models/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

save_path = "./tmp_diffuser_plugin/change_attn/"
style_image = ""
base_image = ""

prompt = ""
negative_prompt = ""
num_ddim_steps = 50

inversion = Inversion(pipe, num_ddim_steps)

prompt = "a cat is lying on the floor"  # a cat is lying on the floor
image_name = "./tmp_example_img/lighting/merge.png"
original_image = Image.open(image_name).convert("RGB")

# process image
image_gt = original_image.resize((1024, 1024))  # why 1024?
image_gt = np.stack([np.array(image_gt)])
prompts = prompt  # 2
# 02: invert
_, x_t, x_stars, prompt_embeds, pooled_prompt_embeds = inversion.invert(image_gt, "", inv_batch_size=1)

images = pipe(prompt=prompt,
              latents=x_stars[-31], x_stars=x_stars,
              negative_prompt_embeds=prompt_embeds,
              negative_pooled_prompt_embeds=pooled_prompt_embeds,
              use_inf_negative_prompt=use_inf_negative_prompt,
              use_advanced_sampling=use_advanced_sampling,
              target_prompt=inf_prompt,
              )
