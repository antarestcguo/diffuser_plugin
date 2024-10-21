import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from plugin_modules.ip_adapter import IPAdapterXL, IPAdapter
from plugin_modules.ip_adapter.gen_baseword import create_maxtext_image_RGB, paste_maxtext, compute_complex_word
from plugin_modules.ip_adapter.word_art_config_dict import word_dict, style_config_dict
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline, \
    StableDiffusionXLControlNetPipeline, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from plugin_modules.ip_adapter.gen_baseword_v2 import image_draw_color
from PIL import Image
import argparse
from random import choice, shuffle

parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int)
parser.add_argument("--end_idx", type=int)
parser.add_argument("--num_inference_steps", type=int, default=20)
args = parser.parse_args()

base_model_path = "./models/stable-diffusion-xl-base-1.0"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
device = "cuda"
controlnet_file_path = "/data/tc_guo/models/QRcodeCN"
controlnet = ControlNetModel.from_pretrained(controlnet_file_path, torch_dtype=torch.float16)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
    controlnet=controlnet,
)
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()
# load ip-adapter
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# load refine model
pipe_refine =StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
    controlnet=controlnet,
)
pipe_refine.enable_vae_slicing()
pipe_refine.enable_xformers_memory_efficient_attention()
# load ip-adapter
ip_model_refine = IPAdapterXL(pipe_refine, image_encoder_path, ip_ckpt, device)

save_path = "./tmp_ip_adpater_qrcode_control_imageinput"
if not os.path.exists(save_path):
    os.makedirs(save_path)

style_path = "./tmp_example_img/style_imgs"
font_file_dict = {
    'baoli': {"file_path": './resource/font/STBaoliSC-Regular-01.ttf', "font_size": 812},
    # 'Arial': './resource/font/ArialNarrowBold.ttf',
    # 'tetai': {"file_path": './resource/font/tetai-2.ttf', "font_size": 595},

    # 'RuiZiYunxingkai': {"file_path": "./resource/font/RuiZiYunZiKuXingKaiGB-2.ttf", "font_size": 812, },
    # 'HanYixingkai': {"file_path": "./resource/font/HanYiXingKaiJian-1.ttf", "font_size": 812, },
    # 'JiZiJingDianxingkai': {
    #     "file_path": "./resource/font/JiZiJingDianXingKaiJianFan-Shan(GEETYPE-XingKaiGBT-Flash)-2.ttf",
    #     "font_size": 812, },
    'QingNiaoxingkai': {"file_path": "./resource/font/QingNiaoHuaGuangXingKai-2.ttf", "font_size": 812, }
}

word_list = ["./tmp_example_img/word_imgs/热爱可抵岁月漫长.png",
             "./tmp_example_img/word_imgs/baidu.jpg",
             "./tmp_example_img/word_imgs/haizei.jpg",
             "./tmp_example_img/word_imgs/male.jpg",
             "./tmp_example_img/word_imgs/xinlang.jpg",
             ]

for i, sub_style_folder in enumerate(list(style_config_dict.keys())):
    if i >= args.start_idx and i <= args.end_idx:
        style_folder = os.path.join(style_path, sub_style_folder)
        style_image_list = os.listdir(style_folder)

        save_sub_path = os.path.join(save_path, sub_style_folder)
        if not os.path.exists(save_sub_path):
            os.makedirs(save_sub_path)

        # gen g_image
        for img_file in word_list:
            style_img = style_image_list[0]
            style_n, style_e = os.path.splitext(style_img)
            try:
                image = Image.open(os.path.join(style_folder, style_img))
            except:
                continue

            # read base image
            ori_img = Image.open(img_file).convert("RGB")
            img_n, img_e = os.path.splitext(img_file.split("/")[-1])

            bg_color = (255, 255, 255)  # style_config_dict[sub_style_folder]['bg_color']
            fg_color = (0, 0, 0)  # style_config_dict[sub_style_folder]['fg_color']
            font_size = style_config_dict[sub_style_folder]['font_size']
            # font_style = style_config_dict[sub_style_folder]['font_style']
            font_style = 'QingNiaoxingkai'  # 'baoli'

            g_image = image_draw_color(ori_img,
                                       bg_color=bg_color,
                                       fg_color=fg_color,
                                       max_side=2048, min_side=512)

            if g_image is None:
                continue
            # select strength strategy
            strength = choice(style_config_dict[sub_style_folder]["strength"])

            images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=args.num_inference_steps,
                                       image=g_image, strength=strength,
                                       prompt=style_config_dict[sub_style_folder]["prompt"],
                                       negative_prompt=style_config_dict[sub_style_folder]["negative_prompt"]
                                       )
            # save image
            for i, it in enumerate(images):
                save_name = os.path.join(save_sub_path,
                                         "%s_%s_%d_strength_%0.3f.jpg" % (style_n, img_n, i, strength))
                it.save(save_name)

            # enhance
            if "enhance" in style_config_dict[sub_style_folder]:
                prompt = style_config_dict[sub_style_folder]['enhance']['prompt']
                negative_prompt = style_config_dict[sub_style_folder]['enhance']['negative_prompt']
                for i, it in enumerate(images):

                    refine_strength = choice(style_config_dict[sub_style_folder]['enhance']['strength'])
                    alpha_ratio = min(style_config_dict[sub_style_folder]['enhance']['alpha_ratio'])


                    refine_image = ip_model_refine.generate(
                        pil_image=image, num_samples=1,
                        num_inference_steps=args.num_inference_steps,
                        image=it,
                        control_image=g_image,
                        strength=refine_strength,
                        prompt=prompt,
                        negative_prompt=negative_prompt)
                    save_name = os.path.join(save_sub_path,
                                             "%s_%s_%d_%0.3f_refinestrength_%0.3f_alpha_%0.3f.jpg" % (
                                                 style_n, img_n, i, strength, refine_strength, alpha_ratio))
                    refine_image[0].save(save_name)
