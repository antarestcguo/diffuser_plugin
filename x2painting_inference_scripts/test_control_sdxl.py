import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from plugin_modules.ip_adapter import IPAdapterXL, IPAdapter
from plugin_modules.ip_adapter.gen_baseword import create_maxtext_image_RGB, paste_maxtext, compute_complex_word
from plugin_modules.ip_adapter.word_art_config_dict import word_dict, style_config_dict
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline, \
    StableDiffusionXLControlNetPipeline, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from PIL import Image
import argparse
from random import choice, shuffle

parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int)
parser.add_argument("--end_idx", type=int)
parser.add_argument("--num_inference_steps", type=int, default=20)
args = parser.parse_args()

base_model_path = "./models/stable-diffusion-xl-base-1.0"
# base_model_path = "/data/tc_guo/models/dreamshaper-xl-1-0"
# base_model_file = "/data/tc_guo/models/sfmodels/samaritan3dCartoon_v40SDXL.safetensors"
base_model_file = "/data/tc_guo/models/sfmodels/dreamshaperXL10_alpha2Xl10.safetensors"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
device = "cuda"
controlnet_file_path = "/data/tc_guo/train_model/x2painting_CN/init/checkpoint-10000/controlnet"
controlnet = ControlNetModel.from_pretrained(controlnet_file_path, torch_dtype=torch.float16)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
    controlnet=controlnet,
)
# pipe = StableDiffusionXLControlNetPipeline.from_single_file(
#     base_model_file,
#     torch_dtype=torch.float16,
#     add_watermarker=False,
#     controlnet=controlnet,
# )
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()
# load ip-adapter
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# load refine model
pipe_refine = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
    controlnet=controlnet,
)
# pipe_refine = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
#     base_model_path,
#     vae=pipe.vae,
#     unet=pipe.unet,
#     torch_dtype=torch.float16,
#     add_watermarker=False,
#     controlnet=controlnet,
# )
pipe_refine.enable_vae_slicing()
pipe_refine.enable_xformers_memory_efficient_attention()
# load ip-adapter
ip_model_refine = IPAdapterXL(pipe_refine, image_encoder_path, ip_ckpt, device)

save_path = "./tmp_diffuser_plugin/control_init_result_iter10k/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

style_path = "./tmp_example_img/style_imgs"
# style_path = "./tmp_ip_adpater_result4shape_base"
font_file_dict = {
    'baoli': {"file_path": './resource/font/STBaoliSC-Regular-01.ttf', "font_size": 812},
    'QingNiaoxingkai': {"file_path": "./resource/font/QingNiaoHuaGuangXingKai-2.ttf", "font_size": 812, }
}

# read hanzi_list
word_list = []
hanzi_file = "./tmp_example_img/test_cn_hanzi.txt"
with open(hanzi_file, 'r') as f:
    for line in f.readlines():
        word_list.append(line.strip())

for i, sub_style_folder in enumerate(list(style_config_dict.keys())):
    if i >= args.start_idx and i <= args.end_idx:
        style_folder = os.path.join(style_path, sub_style_folder)
        style_image_list = os.listdir(style_folder)

        # gen g_image
        for pinyin in word_list:
            save_sub_path = os.path.join(save_path, pinyin + "_" + sub_style_folder)
            if not os.path.exists(save_sub_path):
                os.makedirs(save_sub_path)

            # style_img = choice(style_image_list)
            style_img = style_image_list[0]
            style_n, style_e = os.path.splitext(style_img)
            try:
                image = Image.open(os.path.join(style_folder, style_img))
            except:
                continue

            bg_color = (255, 255, 255)  # style_config_dict[sub_style_folder]['bg_color']
            fg_color = (0, 0, 0)  # style_config_dict[sub_style_folder]['fg_color']
            font_size = style_config_dict[sub_style_folder]['font_size']
            # font_style = style_config_dict[sub_style_folder]['font_style']
            font_style = 'QingNiaoxingkai'  # 'baoli'

            white_ratio, black_ratio = compute_complex_word(pinyin, font_file_dict['baoli']['file_path'],
                                                            font_size=812)
            if white_ratio > 0.41:
                b_complexity = True
            else:
                b_complexity = False

            g_image, select_font_path = create_maxtext_image_RGB(pinyin,
                                                                 bg_color=bg_color,
                                                                 fg_color=fg_color,
                                                                 font_path=font_file_dict[font_style]['file_path'],
                                                                 font_size=font_size)

            if g_image is None:
                continue
            # select strength strategy
            if b_complexity:
                strength = min(style_config_dict[sub_style_folder]["strength"])
            else:
                strength = choice(style_config_dict[sub_style_folder]["strength"])

            images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=args.num_inference_steps,
                                       image=g_image,
                                       # strength=strength,
                                       prompt=style_config_dict[sub_style_folder]["prompt"],
                                       negative_prompt=style_config_dict[sub_style_folder]["negative_prompt"]
                                       )
            # save image
            for i, it in enumerate(images):
                save_name = os.path.join(save_sub_path,
                                         "%s_%s_%d_strength_%0.3f.jpg" % (style_n, pinyin, i, strength))
                it.save(save_name)

            # enhance
            if "enhance" in style_config_dict[sub_style_folder]:
                prompt = style_config_dict[sub_style_folder]['enhance']['prompt']
                negative_prompt = style_config_dict[sub_style_folder]['enhance']['negative_prompt']
                for i, it in enumerate(images):
                    if b_complexity:
                        refine_strength = choice(style_config_dict[sub_style_folder]['enhance']['strength'])
                        alpha_ratio = max(style_config_dict[sub_style_folder]['enhance']['alpha_ratio'])
                    else:
                        refine_strength = choice(style_config_dict[sub_style_folder]['enhance']['strength'])
                        alpha_ratio = min(style_config_dict[sub_style_folder]['enhance']['alpha_ratio'])

                    # enhance_img, pre_enhance = paste_maxtext(pinyin, it, fg_color, bg_color,
                    #                                          select_font_path,
                    #                                          font_size=812,
                    #                                          alpha_ratio=alpha_ratio,
                    #                                          b_bg=b_complexity)

                    refine_image = ip_model_refine.generate(
                        pil_image=image, num_samples=1,
                        control_image=g_image,
                        num_inference_steps=args.num_inference_steps,
                        image=it,
                        strength=refine_strength,
                        controlnet_conditioning_scale=1.0,
                        prompt=prompt,
                        negative_prompt=negative_prompt)
                    save_name = os.path.join(save_sub_path,
                                             "%s_%s_%d_%0.3f_refinestrength_%0.3f_alpha_%0.3f.jpg" % (
                                                 style_n, pinyin, i, strength, refine_strength, alpha_ratio))
                    refine_image[0].save(save_name)
