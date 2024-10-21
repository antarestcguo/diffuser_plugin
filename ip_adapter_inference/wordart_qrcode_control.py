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
parser.add_argument("--num_inference_steps", type=int, default=20)
args = parser.parse_args()

base_model_path = "./models/stable-diffusion-xl-base-1.0"
base_model_file = "/data/tc_guo/models/sfmodels/dreamshaperXL10_alpha2Xl10.safetensors"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
device = "cuda"
controlnet_file_path = "/data/tc_guo/models/QRcodeCN"
controlnet = ControlNetModel.from_pretrained(controlnet_file_path, torch_dtype=torch.float16)

# pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     add_watermarker=False,
#     controlnet=controlnet,
# )
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    controlnet=controlnet,
)
# from_pretrained(
#     base_model_path,
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
    unet=pipe.unet,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    controlnet=controlnet,
)
# from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     add_watermarker=False,
#     controlnet=controlnet,
# )
pipe_refine.enable_vae_slicing()
pipe_refine.enable_xformers_memory_efficient_attention()
# load ip-adapter
ip_model_refine = IPAdapterXL(pipe_refine, image_encoder_path, ip_ckpt, device)

save_path = "./tmp_qrcode_control"
if not os.path.exists(save_path):
    os.makedirs(save_path)

style_path = "./tmp_example_img/style_imgs"
font_file_dict = {
    'baoli': {"file_path": './resource/font/STBaoliSC-Regular-01.ttf', "font_size": 812},
    'QingNiaoxingkai': {"file_path": "./resource/font/QingNiaoHuaGuangXingKai-2.ttf", "font_size": 812, }
}

# read hanzi_list
word_list = []
word_list.append("刘")
hanzi_file = "./tmp_example_img/select_hanzi.txt"
with open(hanzi_file, 'r') as f:
    for line in f.readlines():
        word_list.append(line.strip())

negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, low quality, long neck, frame, text, worst quality,watermark, deformed, ugly,  blur, out of focus,extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs,"
prompt_dict = {
    "profilephoto":"a girl,solo,long_hair,looking at viewer,black hair,upper body,lips,profilephoto, hdr, detailed, photographic",
    "cartoon":"(1 girl:1.3),(pink dress:1.3), upper body, white hair, from side, decorative designs, (leaf:1.6),BREAK, (sunset background:1.3)"
}
for save_key,prompt in prompt_dict.items():
    # gen g_image
    for pinyin in word_list[:5]:
        save_sub_path = os.path.join(save_path, save_key)
        if not os.path.exists(save_sub_path):
            os.makedirs(save_sub_path)

        bg_color = (255, 255, 255)  # style_config_dict[sub_style_folder]['bg_color']
        fg_color = (0, 0, 0)  # style_config_dict[sub_style_folder]['fg_color']
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
                                                             font_size=812)

        if g_image is None:
            continue
        g_image.save(os.path.join(save_sub_path,"base.jpg"))
        images = pipe(num_samples=4, num_inference_steps=args.num_inference_steps,
                                   image=g_image,
                                   prompt=prompt,
                                   negative_prompt=negative_prompt
                                   ).images
        # save image
        for i, it in enumerate(images):
            save_name = os.path.join(save_sub_path,
                                     "loop1_%s_%d.jpg" % ( pinyin,i))
            it.save(save_name)

        # enhance
        for i, it in enumerate(images):
            refine_strength = 0.7
            refine_image = pipe_refine(
                num_samples=1,
                # control_image=g_image,
                num_inference_steps=args.num_inference_steps,
                image=it,
                strength=refine_strength,
                prompt=prompt,
                negative_prompt=negative_prompt).images
            save_name = os.path.join(save_sub_path,
                                     "refine_%s_%0.3f_%d.jpg" % (
                                         pinyin,refine_strength, i))
            refine_image[0].save(save_name)
