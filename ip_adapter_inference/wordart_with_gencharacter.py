import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from plugin_modules.ip_adapter import IPAdapterXL, IPAdapter
from plugin_modules.ip_adapter.gen_baseword import create_text_image_RGB
from plugin_modules.ip_adapter.word_art_config_dict import word_dict, style_config_dict
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int)
parser.add_argument("--end_idx", type=int)
args = parser.parse_args()

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

save_path = "./tmp_ip_adpater_result_select_strength"
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

for i, sub_style_folder in enumerate(list(style_config_dict.keys())):
    if i >= args.start_idx and i <= args.end_idx:
        style_folder = os.path.join(style_path, sub_style_folder)
        style_image_list = os.listdir(style_folder)

        for style_img in style_image_list:
            style_n, style_e = os.path.splitext(style_img)
            try:
                image = Image.open(os.path.join(style_folder, style_img))
            except:
                continue

            save_sub_path = os.path.join(save_path, sub_style_folder + "_" + style_n)
            if not os.path.exists(save_sub_path):
                os.makedirs(save_sub_path)
            # gen g_image
            for pinyin in word_dict.keys():
                bg_color = style_config_dict[sub_style_folder]['bg_color']
                fg_color = style_config_dict[sub_style_folder]['fg_color']
                font_size = style_config_dict[sub_style_folder]['font_size']
                font_style = style_config_dict[sub_style_folder]['font_style']

                g_image = create_text_image_RGB(word_dict[pinyin],
                                                bg_color=bg_color,
                                                fg_color=fg_color,
                                                font_path=font_file_dict[font_style]['file_path'],
                                                font_size=font_size)
                # save g_image
                save_name = os.path.join(save_sub_path, "base_%s_%s.jpg" % (font_style, pinyin))
                g_image.save(save_name)
                for strength in style_config_dict[sub_style_folder]["strength"]:
                    images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50,
                                               image=g_image, strength=strength,
                                               prompt=style_config_dict[sub_style_folder]["prompt"],
                                               negative_prompt=style_config_dict[sub_style_folder]["negative_prompt"]
                                               )
                    # save image
                    for i, it in enumerate(images):
                        save_name = os.path.join(save_sub_path, "%s_%d_strength_%f.jpg" % (pinyin, i, strength))
                        it.save(save_name)
