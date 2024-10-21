import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from plugin_modules.ip_adapter import IPAdapterXL, IPAdapter
from plugin_modules.ip_adapter.gen_basewordimg_v2 import gen_character, is_chinese, gen_word_EN, gen_word_CN_together, \
    b_complex_CN, enhance_character, gen_word_CN_separate, gen_word_CN_multidiffusion
from plugin_modules.ip_adapter.word_art_config_dict import word_dict, style_config_dict

from plugin_modules.ip_adapter.enhance_pipeline.SDImg2Img_multidiffusion import StableDiffusionXLImg2ImgPanoramaPipeline

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline, \
    UNet2DConditionModel, EulerDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, \
    TCDScheduler
from PIL import Image
import argparse
from random import choice, shuffle
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int)
parser.add_argument("--end_idx", type=int)
parser.add_argument("--num_inference_steps", type=int, default=20)
args = parser.parse_args()

base_model_path = "./models/stable-diffusion-xl-base-1.0"
base_model_file = "/data/tc_guo/models/sfmodels/dreamshaperXL_lightningDPMSDE.safetensors"
# base_model_file = "/data/tc_guo/models/sdxl_lighting_bytedance/sdxl_lightning_8step.safetensors"
# base_model_file = "/data/tc_guo/models/sdxl_lighting_bytedance/Hyper-SDXL-8steps-lora.safetensors"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
device = "cuda"

save_path = "./tmp_ip_adpater_v11_newstyle"
if not os.path.exists(save_path):
    os.makedirs(save_path)

base_pipe = StableDiffusionXLImg2ImgPanoramaPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)
base_pipe.enable_vae_slicing()
base_pipe.enable_xformers_memory_efficient_attention()

ip_base_model = IPAdapterXL(base_pipe, image_encoder_path, ip_ckpt, device)

pipe = StableDiffusionXLImg2ImgPanoramaPipeline.from_single_file(
    base_model_file,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
# pipe.scheduler = TCDScheduler.from_config(
#     pipe.scheduler.config,
#     # use_karras_sigmas=True,
# )
# pipe = StableDiffusionXLPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16, variant="fp16").to("cuda")
# pipe.load_lora_weights(base_model_file)
# pipe.fuse_lora()
# pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
# guidance_scale = 0
# eta = 1.0

pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()
# load ip-adapter
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

style_path = "./tmp_example_img/style_imgs"

CN_font_file_dict = {
    # 'baoli': {"file_path": './resource/font/STBaoliSC-Regular-01.ttf', "font_size": 812},
    # 'Arial': './resource/font/ArialNarrowBold.ttf',
    # 'tetai': {"file_path": './resource/font/tetai-2.ttf', "font_size": 595},

    # 'RuiZiYunxingkai': {"file_path": "./resource/font/RuiZiYunZiKuXingKaiGB-2.ttf", "font_size": 812, },
    # 'HanYixingkai': {"file_path": "./resource/font/HanYiXingKaiJian-1.ttf", "font_size": 812, },
    # 'JiZiJingDianxingkai': {
    #     "file_path": "./resource/font/JiZiJingDianXingKaiJianFan-Shan(GEETYPE-XingKaiGBT-Flash)-2.ttf",
    #     "font_size": 812, },
    'QingNiaoxingkai': {"file_path": "./resource/font/QingNiaoHuaGuangXingKai-2.ttf", "font_size": 812, }
}
EN_font_file_dict = {
    # "Monster": {"file_path": "./resource/font/A-Little-Monster-2.ttf", "font_size": 812},
    "HFPuff": {"file_path": "./resource/font/HFPuff-2.ttf", "font_size": 812},
    # "HFShinySunday": {"file_path": "./resource/font/HFShinySunday-2.ttf", "font_size": 812},

    # "HFSoda": {"file_path": "./resource/font/HFSoda-2.ttf", "font_size": 812},
    # "justanotherhand": {"file_path": "./resource/font/justanotherhand-regular-2.ttf", "font_size": 812},
}
EN_default_font_path = ['./resource/font/ArialNarrowBold.ttf']
CN_default_font_path = ['./resource/font/STBaoliSC-Regular-01.ttf', './resource/font/ArialNarrowBold.ttf']

# read hanzi_list
# word_list = ["热爱", "喜乐"]
word_list = []

file_name_list = [
    # "./tmp_example_img/mix_EN_word.txt",
    # "./tmp_example_img/mix_CN_word.txt",
    "./tmp_example_img/mix_CN_character.txt",
    "./tmp_example_img/mix_EN_character.txt",

]
for it_file in file_name_list:
    with open(it_file, 'r') as f:
        for line in f.readlines():
            word_list.append(line.strip())

for i, sub_style_folder in enumerate(list(style_config_dict.keys())):
    if i >= args.start_idx and i <= args.end_idx:
        style_folder = os.path.join(style_path, sub_style_folder)
        style_image_list = os.listdir(style_folder)

        # gen g_image
        for pinyin in word_list:

            save_sub_path = os.path.join(save_path, sub_style_folder)
            if not os.path.exists(save_sub_path):
                os.makedirs(save_sub_path)

            style_img = choice(style_image_list)
            style_n, style_e = os.path.splitext(style_img)
            try:
                image = Image.open(os.path.join(style_folder, style_img))
            except:
                continue

            bg_color = style_config_dict[sub_style_folder]['bg_color']
            fg_color = style_config_dict[sub_style_folder]['fg_color']
            font_size = style_config_dict[sub_style_folder]['font_size']

            # gen base image
            text_to_draw = pinyin.strip().split(' ')[0]
            if len(text_to_draw) == 1:
                # character
                if is_chinese(text_to_draw):
                    default_font_path_list = CN_default_font_path
                    font_path = CN_font_file_dict['QingNiaoxingkai']['file_path']
                    base_img = gen_character(text_to_draw, font_path, default_font_path_list, fg_color, bg_color)
                    # b_complex character
                    b_complex = b_complex_CN(text_to_draw, font_path, default_font_path_list, font_size=812,
                                             text_ratio=0.85)
                    if b_complex:
                        strength = min(style_config_dict[sub_style_folder]['strength'])
                    else:
                        strength = choice(style_config_dict[sub_style_folder]['strength'])
                else:
                    default_font_path_list = EN_default_font_path
                    font_path = EN_font_file_dict['HFPuff']['file_path']
                    base_img = gen_character(text_to_draw, font_path, default_font_path_list, fg_color, bg_color)
                    strength = choice(style_config_dict[sub_style_folder]['EN_strength'])
                    b_complex = False

                images = ip_model.generate(
                    pil_image=image, num_samples=4, num_inference_steps=8,
                    image=base_img, strength=strength,
                    prompt=style_config_dict[sub_style_folder]["prompt"],
                    negative_prompt=style_config_dict[sub_style_folder]["negative_prompt"],
                )
            else:
                if text_to_draw.encode('utf-8').isalpha():  # not used now
                    default_font_path_list = EN_default_font_path
                    font_path = EN_font_file_dict['HFPuff']['file_path']
                    base_img = gen_word_EN(text_to_draw, font_path, default_font_path_list, fg_color, bg_color)
                    strength = choice(style_config_dict[sub_style_folder]['EN_strength'])
                else:  # test here
                    text_to_draw = text_to_draw[:4]
                    default_font_path_list = CN_default_font_path
                    font_path = CN_font_file_dict['QingNiaoxingkai']['file_path']
                    strength = choice(style_config_dict[sub_style_folder]['strength'])  # + 0.05

                    # base_img = gen_word_CN_together(text_to_draw, font_path, default_font_path_list, fg_color,
                    #                                 bg_color)

                    final_img, slice_view_list, crop_list, text_ratio_x, text_ratio_y, resize_offset_x = gen_word_CN_multidiffusion(
                        text_to_draw,
                        font_path, default_font_path_list, fg_color,
                        bg_color)

                    images = ip_model.multidiffusion_generate(pil_image=image,
                                                              num_samples=4,
                                                              num_inference_steps=args.num_inference_steps,
                                                              image=final_img,
                                                              strength=strength,
                                                              prompt=style_config_dict[sub_style_folder]["prompt"],
                                                              negative_prompt=style_config_dict[sub_style_folder][
                                                                  "negative_prompt"],
                                                              views=slice_view_list,
                                                              )
                    # images = []
                    # for it in tmp_images:
                    #     # add another img2img
                    #     it_images = ip_model.generate(
                    #         pil_image=image, num_samples=1,
                    #         num_inference_steps=args.num_inference_steps,
                    #         image=it,
                    #         strength=strength,
                    #         prompt=style_config_dict[sub_style_folder]["prompt"],
                    #         negative_prompt=style_config_dict[sub_style_folder][
                    #                                          "negative_prompt"])
                    #     import pdb
                    #     pdb.set_trace()
                    #     images.append(it_images[0])
                    #     a = 0

            # save image
            for i, it in enumerate(images):
                save_name = os.path.join(save_sub_path,
                                         "%s_%s_%d_strength_%0.3f.jpg" % (
                                             text_to_draw, style_n, i, strength))
                it.save(save_name)

            # enhance
            if "enhance" in style_config_dict[sub_style_folder]:
                prompt = style_config_dict[sub_style_folder]['enhance']['prompt']
                negative_prompt = style_config_dict[sub_style_folder]['enhance']['negative_prompt']
                for i, it in enumerate(images):
                    # check CN/EN char/word
                    if len(text_to_draw) == 1:
                        if is_chinese(text_to_draw):
                            default_font_path_list = CN_default_font_path
                            font_path = CN_font_file_dict['QingNiaoxingkai']['file_path']
                            refine_strength = choice(style_config_dict[sub_style_folder]['enhance']['strength'])
                            if b_complex:
                                alpha_ratio = max(style_config_dict[sub_style_folder]['enhance']['alpha_ratio'])
                            else:
                                alpha_ratio = choice(style_config_dict[sub_style_folder]['enhance']['alpha_ratio'])
                        else:
                            alpha_ratio = 0
                            refine_strength = choice(style_config_dict[sub_style_folder]['enhance']['strength']) + 0.1

                        enhance_img, pre_enhance = enhance_character(text_to_draw, it, font_path,
                                                                     default_font_path_list, fg_color, bg_color,
                                                                     alpha_ratio=alpha_ratio, b_bg=b_complex)

                        refine_image = ip_base_model.generate(
                            pil_image=image, num_samples=1,
                            num_inference_steps=args.num_inference_steps,
                            image=enhance_img,
                            strength=refine_strength,
                            prompt=prompt,
                            negative_prompt=negative_prompt)
                        save_name = os.path.join(save_sub_path,
                                                 "refine_%s_%s_%d_%0.3f_refinestrength_%0.3f.jpg" % (
                                                     text_to_draw, style_n, i, strength, refine_strength))
                        refine_image[0].save(save_name)

                        save_name = os.path.join(save_sub_path,
                                                 "prerefine_%s_%s_%d_%0.3f_refinestrength_%0.3f.jpg" % (
                                                     text_to_draw, style_n, i, strength, refine_strength))
                        enhance_img.save(save_name)

                print("gen end", "-" * 30, pinyin)
