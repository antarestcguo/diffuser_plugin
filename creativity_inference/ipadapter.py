import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))
from plugin_modules.ip_adapter.ip_adapter import IPAdapterXL, IPAdapterPlusXL
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline, \
    StableDiffusionXLControlNetPipeline, ControlNetModel
# control inpainting, control img2img,
import torch
from PIL import Image
import cv2
import numpy as np

negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, low quality, long neck, frame, text, worst quality,watermark, deformed, ugly,  blur, out of focus,extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, "

device = "cuda"
save_path = "./tmp_diffuser_plugin/tmp_c_result/wolf"
if not os.path.exists(save_path):
    os.makedirs(save_path)
# define image and prompt
# ref_image_name = "./tmp_resource/ipadapter_resource/snowleopard.jpeg"
# ref_image_name = "./tmp_example_img/ipadater_img/sheli_head.jpeg"
# ref_image_name = "./tmp_example_img/instant_style/cartooncaracal.jpeg"
ref_image_name = "./tmp_diffuser_plugin/tmp_c_result/wolf/2_cannycontrol_frame10.jpg"
canny_image_name = "./tmp_example_img/ipadater_img/wolf/canny_110.jpg"
prompt = "a wolf walk on the loess plateau, cartoon style, best quality, masterpiece"

# base model
# base_model_file = "/data/tc_guo/models/sfmodels/samaritan3dCartoon_v40SDXL.safetensors"
# base_model_file = "/data/tc_guo/models/sfmodels/dreamshaperXL10_alpha2Xl10.safetensors"
# base_model_file = "/data/tc_guo/models/sfmodels/Juggernaut.safetensors"
base_model_file = "/data/tc_guo/models/sfmodels/animagine-xl-3.1.safetensors"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
# xlplus
# image_encoder_refine_path = "./models/IP-Adapter/models/image_encoder"
# ip_ckpt_refine = "./models/IP-Adapter/sdxl_model/ip-adapter-plus_sdxl_vit-h.safetensors"

pipe_type = "canny_control"  # img2img, canny_control,XL

if pipe_type == "XL":
    pipe = StableDiffusionXLPipeline.from_single_file(
        base_model_file,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
elif pipe_type == "img2img":
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        base_model_file,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
elif pipe_type == "canny_control":
    controlnet_path = "./models/instantID_model/diffusers/controlnet-canny-sdxl-1.0"
    # controlnet_path = "/data/tc_guo/models/models--xinsir--controlnet-scribble-sdxl-1.0/snapshots/af8f42dfa5429c2c7a97d5f52a172d637e918969"
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        base_model_file,
        torch_dtype=torch.float16,
        use_safetensors=True,
        controlnet=controlnet,
    )

# lora_path = "/data/tc_guo/models/sfmodels/cartoon_stickers_xl_v1.safetensors"
# pipe.load_lora_weights(lora_path)
# pipe.enable_lora()
# reduce GPU MEM
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=4)  # modify the num_tokens=16

ref_image = Image.open(ref_image_name).convert("RGB")
canny_image = Image.open(canny_image_name).convert("RGB")
# generate
if pipe_type == "XL":
    images = ip_model.generate(pil_image=ref_image, num_samples=4, num_inference_steps=20,
                               prompt=prompt, scale=1.0, negative_prompt=negative_prompt)
elif pipe_type == "img2img":
    images = ip_model.generate(pil_image=ref_image,image=ref_image, num_samples=4, num_inference_steps=20,
                               prompt=prompt, scale=1.0, negative_prompt=negative_prompt,strength=0.7)
elif pipe_type == "canny_control":
    t1 = 80
    t2 = 200
    # image = cv2.cvtColor(np.array(canny_image), cv2.COLOR_RGB2BGR)
    # edges = cv2.Canny(image, t1, t2)
    # canny_img = Image.fromarray(edges, "L")
    # canny_img.save(os.path.join(save_path,"canny_bear.jpg"))
    images = ip_model.generate(pil_image=ref_image, image=canny_image, num_samples=4, num_inference_steps=20,
                               prompt=prompt, scale=1.0, negative_prompt=negative_prompt)

# save
for i, it in enumerate(images):
    # save_name = os.path.join(save_path, "{}_{}_bearfight.jpg".format(i, pipe_type))
    save_name = os.path.join(save_path, "{}_cannycontrol_frame110.jpg".format(i))
    it.save(save_name)
