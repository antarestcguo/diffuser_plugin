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
wordart_image_path = "./tmp_example_img/wordart_city"

# prompt related
negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, low quality, long neck, frame, text, big, 3d rendering, unreal"

prompt_dict = {
    "city": "a future city, building, road, flyover, cyberpunk,ablaze with lights, masterpiece, photographic, intricate detail, Ultra Detailed hyperrealistic real photo, 8k, high quality, hdr, studio lighting, professional, trending on artstation,",
}
# read image prompt
word_art_list = os.listdir(wordart_image_path)
for word_art_img in word_art_list:
    word_n, word_e = os.path.splitext(word_art_img)
    g_image = Image.open(os.path.join(wordart_image_path, word_art_img))
    if word_n.find("black") != -1:
        img2img_strength = 0.9
    elif word_n.find("grey") != -1:
        img2img_strength = 0.8
    elif word_n.find("white") != -1:
        img2img_strength = 0.6

    for sub_style_folder, style_prompt in prompt_dict.items():
        style_folder = os.path.join(style_path, sub_style_folder)
        style_image_list = os.listdir(style_folder)
        for style_img in style_image_list:
            image = Image.open(os.path.join(style_folder, style_img))
            style_n, style_e = os.path.splitext(style_img)
            if style_n.find("night") == -1:
                prompt = "a future city, building, road, flyover,sense of technology,masterpiece, photographic, intricate detail, Ultra Detailed hyperrealistic real photo, 8k, high quality, hdr, studio lighting, professional, trending on artstation,"
            images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50,
                                       image=g_image, strength=img2img_strength, prompt=style_prompt,
                                       negative_prompt=negative_prompt
                                       )
            for i, it in enumerate(images):
                save_name = os.path.join(save_path, "loop1_%s_%s_%s_%d.jpg" % (sub_style_folder, style_n, word_n, i))
                it.save(save_name)
                # refine_images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50,
                #                                   image=images[i], strength=0.6, prompt=style_prompt,
                #                                   negative_prompt=negative_prompt
                #                                   )
                #
                # for _, it in enumerate(refine_images):
                #     save_name = os.path.join(save_path,
                #                              "loop2_%s_%s_%s_%d.jpg" % (sub_style_folder, style_n, word_n, i))
                #     it.save(save_name)
        print("gen end,", word_art_img, sub_style_folder)
