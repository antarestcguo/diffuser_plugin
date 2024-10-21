import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from plugin_modules.ip_adapter import IPAdapterPlusXL, IPAdapter, IPAdapterXL
from plugin_modules.ip_adapter.gen_baseword import create_maxtext_image_RGB, paste_maxtext, compute_complex_word
from plugin_modules.ip_adapter.word_art_config_dict import word_dict, style_config_dict
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import argparse
from random import choice

parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int)
parser.add_argument("--end_idx", type=int)
parser.add_argument("--num_inference_steps", type=int, default=20)
args = parser.parse_args()

base_model_path = "./models/stable-diffusion-xl-base-1.0"

image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"

image_encoder_refine_path = "./models/IP-Adapter/models/image_encoder"
ip_ckpt_refine = "./models/IP-Adapter/sdxl_model/ip-adapter-plus_sdxl_vit-h.safetensors"
device = "cuda"
# 都使用refine model，细节表现很好，但是strength要重新调整，细节包括背景，多样性降低
# 如果先试用model，后使用refine model，同样需要重新调整

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
# reduce GPU MEM
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()

# load ip-adapter
ip_model_refine = IPAdapterPlusXL(pipe, image_encoder_refine_path, ip_ckpt_refine, device,num_tokens=16) # modify the num_tokens=16

ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

save_path = "./tmp_ip_adpaterplus"
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
    # if sub_style_folder not in [
    #     'shuimodesigner_black',
    #     'fengjingdesigner_black',
    #     'shuimodesigner2_black',
    #     'treedesigner_black'
    # ]:
    #     continue
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
                # font_style = style_config_dict[sub_style_folder]['font_style']
                font_style = 'QingNiaoxingkai'  # 'baoli'

                white_ratio, black_ratio = compute_complex_word(word_dict[pinyin], font_file_dict['baoli']['file_path'],
                                                                font_size=812)
                if white_ratio > 0.41:
                    b_complexity = True
                else:
                    b_complexity = False

                g_image = create_maxtext_image_RGB(word_dict[pinyin],
                                                   bg_color=bg_color,
                                                   fg_color=fg_color,
                                                   font_path=font_file_dict[font_style]['file_path'],
                                                   font_size=font_size)
                # save g_image
                save_name = os.path.join(save_sub_path, "base_%s_%s.jpg" % (font_style, pinyin))
                g_image.save(save_name)

                # select strength strategy
                if b_complexity:
                    strength = min(style_config_dict[sub_style_folder]["strength"])
                else:
                    strength = choice(style_config_dict[sub_style_folder]["strength"])

                images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=args.num_inference_steps,
                                           image=g_image, strength=strength,
                                           prompt=style_config_dict[sub_style_folder]["prompt"],
                                           negative_prompt=style_config_dict[sub_style_folder]["negative_prompt"]
                                           )
                # save image
                for i, it in enumerate(images):
                    save_name = os.path.join(save_sub_path,
                                             "%s_%s_%d_strength_%0.3f.jpg" % (font_style, pinyin, i, strength))
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

                        enhance_img, pre_enhance = paste_maxtext(word_dict[pinyin], it, fg_color, bg_color,
                                                                 font_file_dict[font_style]['file_path'],
                                                                 font_size=812,
                                                                 alpha_ratio=alpha_ratio,
                                                                 b_bg=b_complexity)

                        refine_image = ip_model_refine.generate(
                            pil_image=image, num_samples=1,
                            num_inference_steps=args.num_inference_steps,
                            image=enhance_img,
                            strength=refine_strength,
                            prompt=prompt,
                            negative_prompt=negative_prompt)
                        save_name = os.path.join(save_sub_path,
                                                 "refine_%s_%s_%d_%0.3f_refinestrength_%0.3f_alpha_%0.3f.jpg" % (
                                                     font_style, pinyin, i, strength, refine_strength, alpha_ratio))
                        refine_image[0].save(save_name)

                        save_name = os.path.join(save_sub_path,
                                                 "prerefine_%s_%s_%d_%0.3f_refinestrength_%0.3f_alpha_%0.3f.jpg" % (
                                                     font_style, pinyin, i, strength, refine_strength, alpha_ratio))
                        enhance_img.save(save_name)
