import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from plugin_modules.ip_adapter import IPAdapterXL
from plugin_modules.ip_adapter.gen_basewordimg_v2_clean import is_chinese, gen_character, gen_word_CN, \
    enhance_character, enhance_word_CN, gen_word_EN

from plugin_modules.ip_adapter.word_art_config_dict import style_config_dict

from plugin_modules.ip_adapter.enhance_pipeline.SDImg2Img_multidiffusion import StableDiffusionXLImg2ImgPanoramaPipeline

import torch

from PIL import Image
import argparse
from random import choice, shuffle

parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int)
parser.add_argument("--end_idx", type=int)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--fast_num_inference_steps", type=int, default=8)
args = parser.parse_args()

base_model_path = "./models/stable-diffusion-xl-base-1.0"
base_model_file = "/data/tc_guo/models/sfmodels/dreamshaperXL_lightningDPMSDE.safetensors"

image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
device = "cuda"

save_path = "./tmp_ip_adpater_multicha_mix"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load base model
print("load base XL pipeline")
base_pipe = StableDiffusionXLImg2ImgPanoramaPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)
base_pipe.enable_vae_slicing()
base_pipe.enable_xformers_memory_efficient_attention()
ip_base_model = IPAdapterXL(base_pipe, image_encoder_path, ip_ckpt, device)

# load lightning model
print("load lightning XL pipeline")
pipe = StableDiffusionXLImg2ImgPanoramaPipeline.from_single_file(
    base_model_file,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

style_path = "./tmp_example_img/style_imgs"

CN_font_file_dict = {
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
EN_font_file_dict = {
    # "Monster": {"file_path": "./resource/font/A-Little-Monster-2.ttf", "font_size": 812},
    "HFPuff": {"file_path": "./resource/font/HFPuff-2.ttf", "font_size": 812},
    # "HFShinySunday": {"file_path": "./resource/font/HFShinySunday-2.ttf", "font_size": 812},

    # "HFSoda": {"file_path": "./resource/font/HFSoda-2.ttf", "font_size": 812},
    # "justanotherhand": {"file_path": "./resource/font/justanotherhand-regular-2.ttf", "font_size": 812},
}
EN_default_font_path = ['./resource/font/ArialNarrowBold.ttf']
CN_default_font_path = ['./resource/font/STBaoliSC-Regular-01.ttf', './resource/font/ArialNarrowBold.ttf']

word_list = []
word_list.append("赵")
word_list.append("刘")
word_list.append("魏")
word_list.append("李")
word_list.append("钱")
file_name_list = [
    # "./tmp_example_img/mix_EN_word.txt",
    # "./tmp_example_img/mix_CN_word.txt",
    # "./tmp_example_img/mix_CN_character.txt",
    # "./tmp_example_img/mix_EN_character.txt",
    "./tmp_example_img/test_example.txt",

]
for it_file in file_name_list:
    with open(it_file, 'r') as f:
        for line in f.readlines():
            word_list.append(line.strip())

for i, sub_style_folder in enumerate(list(style_config_dict.keys())):
    if sub_style_folder != "girl_real":
        continue
    if sub_style_folder in ["butterflygirl_black", "cartoongirl2_while"]:
        continue
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
            prompt = style_config_dict[sub_style_folder]["prompt"]
            negative_prompt = style_config_dict[sub_style_folder]["negative_prompt"]
            # gen base image
            text_to_draw = pinyin.strip().split(' ')[0]
            if len(text_to_draw) == 1:
                # character
                char_type = "CN" if is_chinese(text_to_draw) else "EN"

                # get font param
                default_font_path_list = eval(char_type + "_default_font_path")
                font_style = choice(style_config_dict[sub_style_folder][(char_type + "_font_style").replace("CN_", "")])
                font_path = CN_font_file_dict[font_style]['file_path']

                # gen base image
                base_img, resize_crop_grey, \
                    start_x, start_y, b_complex = gen_character(text_to_draw,
                                                                font_path,
                                                                default_font_path_list,
                                                                fg_color, bg_color, char_type)

                # strength
                strength = choice(style_config_dict[sub_style_folder][
                                      (char_type + '_strength').replace("CN_", "")]) if not b_complex else min(
                    style_config_dict[sub_style_folder][(char_type + '_strength').replace("CN_", "")])

                # gen image loop1
                images = ip_base_model.generate(pil_image=image,
                                                num_samples=4, num_inference_steps=args.num_inference_steps,
                                                image=base_img, strength=strength,
                                                prompt=prompt,
                                                negative_prompt=negative_prompt
                                                )
                debug_images = []
            else:  # multi char
                word_type = "EN" if text_to_draw.encode('utf-8').isalpha() else "CN"
                # get font param
                default_font_path_list = eval(word_type + "_default_font_path")
                font_style = choice(style_config_dict[sub_style_folder][(word_type + "_font_style").replace("CN_", "")])
                font_path = eval(word_type + "_font_file_dict")[font_style]['file_path']

                # gen base image
                final_img, views, resize_crop_list, start_x, start_y, b_complex_list = \
                    eval("gen_word_" + word_type)(
                        text_to_draw,
                        font_path, default_font_path_list, fg_color,
                        bg_color)
                strength_template = style_config_dict[sub_style_folder][
                    (word_type + '_strength').replace("CN_", "")]
                choice_strength = choice(strength_template)
                strength_list = [min(strength_template) if it else choice_strength for it in b_complex_list]

                images, debug_images = \
                    ip_model.word_generate(pil_image=image,
                                           views=views,
                                           full_image=final_img,
                                           num_samples=4, num_inference_steps=args.fast_num_inference_steps,
                                           strength_list=strength_list,
                                           prompt=prompt,
                                           negative_prompt=negative_prompt,
                                           # debug=True,
                                           # seed=1024,
                                           )
            # save first loop
            save_strength = strength if len(text_to_draw) == 1 else strength_list[0]
            for i, it in enumerate(images):
                save_name = os.path.join(save_sub_path,
                                         "%s_%s_%d_strength_%0.3f.jpg" % (
                                             text_to_draw, style_n, i, save_strength))
                it.save(save_name)
            # debug save patch
            for i, it in enumerate(debug_images):
                save_name = os.path.join(save_sub_path,
                                         "debug_%s_%s_%d_strength_%0.3f.jpg" % (
                                             text_to_draw, style_n, i, save_strength))
                it.save(save_name)

            # enhance
            if "enhance" in style_config_dict[sub_style_folder]:
                prompt = style_config_dict[sub_style_folder]['enhance']['prompt']
                negative_prompt = style_config_dict[sub_style_folder]['enhance']['negative_prompt']
                for i, it in enumerate(images):
                    if len(text_to_draw) == 1:
                        refine_strength = choice(style_config_dict[sub_style_folder]['enhance']['strength'])

                        if char_type == "EN":
                            alpha_ratio = 0
                            refine_strength += 0.1
                        else:
                            alpha_ratio = max(
                                style_config_dict[sub_style_folder]['enhance']['alpha_ratio']) if b_complex else choice(
                                style_config_dict[sub_style_folder]['enhance']['alpha_ratio'])

                        enhance_img, pre_enhance = enhance_character(it,
                                                                     resize_crop_grey, start_x, start_y, fg_color,
                                                                     bg_color, alpha_ratio,
                                                                     b_bg=b_complex)

                        refine_image = ip_base_model.generate(
                            pil_image=image, num_samples=1,
                            num_inference_steps=args.num_inference_steps,
                            image=enhance_img,
                            strength=refine_strength,
                            prompt=prompt,
                            negative_prompt=negative_prompt)[0]
                    else:
                        alpha_template = style_config_dict[sub_style_folder]['enhance'][
                            'word_' + word_type + '_alpha_ratio']
                        choice_alpha = choice(alpha_template)
                        alpha_ratio_list = [max(alpha_template) if it_complex else choice_alpha for it_complex in
                                            b_complex_list]
                        refine_strength = choice(
                            style_config_dict[sub_style_folder]['enhance']['word_' + word_type + '_strength'])

                        if word_type == "EN":
                            enhance_img = it

                        else:
                            enhance_img, pre_enhance = enhance_word_CN(it,
                                                                       resize_crop_list, start_x, start_y, fg_color,
                                                                       bg_color, alpha_ratio_list, b_complex_list)

                        # if len(views) == 0:
                        #     refine_image = ip_base_model.generate(
                        #         pil_image=image, num_samples=1,
                        #         num_inference_steps=args.num_inference_steps,
                        #         image=enhance_img,
                        #         strength=refine_strength,
                        #         prompt=prompt,
                        #         negative_prompt=negative_prompt)[0]
                        # else:

                        refine_image = ip_base_model.multidiffusion_generate(
                            pil_image=image, num_samples=1, views=views,
                            num_inference_steps=args.num_inference_steps,
                            image=enhance_img,
                            strength=refine_strength,
                            prompt=prompt,
                            negative_prompt=negative_prompt)[0]

                    # save image
                    save_name = os.path.join(
                        save_sub_path, "refine_%s_%s_%d_%0.3f_refinestrength_%0.3f.jpg" % (
                            text_to_draw, style_n, i, save_strength, refine_strength))
                    refine_image.save(save_name)

                    # save_name = os.path.join(
                    #     save_sub_path, "prerefine_%s_%s_%d_%0.3f_refinestrength_%0.3f.jpg" % (
                    #         text_to_draw, style_n, i, save_strength, refine_strength))
                    # enhance_img.save(save_name)

            print("gen end", "-" * 30, pinyin)
