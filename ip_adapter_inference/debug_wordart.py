import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, \
    DDIMScheduler, AutoencoderKL
from PIL import Image

from plugin_modules.ip_adapter import IPAdapterXL, IPAdapter

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image

base_model_path = "./models/stable-diffusion-xl-base-1.0"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
device = "cuda"
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
# load ip-adapter
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

save_path = "./tmp_ip_adpater_result"
if not os.path.exists(save_path):
    os.makedirs(save_path)

style_path = "./tmp_example_img/style_imgs"
wordart_image_path = "./tmp_example_img/wordart_black"

# prompt related
negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, low quality, long neck, frame, text, big, 3d rendering, unreal"

prompt_dict = {
    "butterflygirl_black": "a cartoon style gile with Long flowing black hair, blue butterfly on hair, full body photo, full-length portrait, watercolor,ink and wash, light clean background",
    # "cat_black": "a Tabby cat full body photo, cartoon watercolor, light clean background",
    # "dragon_black": "a cartoon style Martial arts Style person with while Chinese Dragon,full body photo, full-length portrait, light clean background ",
    "male_black": "a cartoon style Martial arts Style man, male, in the bamboo forest, ,full body photo, full-length portrait, light clean background ",
    "tree_black": "log cabin with trees, lake, mountain, wooden boat, ink and wash, watercolor",
    "winter_black": "log cabin with trees in snow, lake, mountain, plum blossom, wooden boat, ink and wash, watercolor",

}
# read image prompt
word_art_list = os.listdir(wordart_image_path)
for word_art_img in word_art_list:
    word_n, word_e = os.path.splitext(word_art_img)
    g_image = Image.open(os.path.join(wordart_image_path, word_art_img))
    for sub_style_folder, style_prompt in prompt_dict.items():
        style_folder = os.path.join(style_path, sub_style_folder)
        style_image_list = os.listdir(style_folder)
        for style_img in style_image_list:
            image = Image.open(os.path.join(style_folder, style_img))
            style_n, style_e = os.path.splitext(style_img)
            images = ip_model.generate(pil_image=[image, image], num_samples=4, num_inference_steps=50,
                                       image=g_image, strength=0.8, prompt=style_prompt, negative_prompt=negative_prompt
                                       )
            for i, it in enumerate(images):
                save_name = os.path.join(save_path, "loop1_%s_%s_%s_%d.jpg" % (sub_style_folder, style_n, word_n, i))
                it.save(save_name)
            #     refine_images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50,
            #                                image=images[i], strength=0.6, prompt=style_prompt,
            #                                negative_prompt=negative_prompt
            #                                )
            #
            #     for _, it in enumerate(refine_images):
            #         save_name = os.path.join(save_path,
            #                                  "loop2_%s_%s_%s_%d.jpg" % (sub_style_folder, style_n, word_n, i))
            #         it.save(save_name)
        print("gen end,", word_art_img, sub_style_folder)
