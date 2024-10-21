import PIL.Image as Image
import sys
import os
import PIL

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))
from plugin_modules.ip_adapter import IPAdapterXL, IPAdapter
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline,StableDiffusionXLControlNetPipeline,ControlNetModel,StableDiffusionXLControlNetImg2ImgPipeline

base_model_path = "./models/stable-diffusion-xl-base-1.0"
# base_model_path = "./models/stable-diffusion-v1-5"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
# vae_model_path = "./models/IP-Adapter/stabilityai/sd-vae-ft-mse"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
device = "cuda"
# controlnet_path = "./models/instantID_model/diffusers/controlnet-depth-sdxl-1.0"
controlnet_path = "./models/instantID_model/diffusers/controlnet-depth-sdxl-1.0-small"


# load SDXL pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                             # variant="fp16",
                                             use_safetensors=True,
                                             torch_dtype=torch.float16).to(device)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    use_safetensors=True,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)

# load ip-adapter
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# read image prompt
image = Image.open("./tmp_example_img/images/futurecity.png")
depth_map = Image.open("./tmp_example_img/structure_controls/city_structure.jpeg").resize((1024, 1024))

num_samples = 2
images = ip_model.generate(pil_image=image, image=depth_map, controlnet_conditioning_scale=0.7, num_samples=num_samples, num_inference_steps=50, seed=42,
                           prompt="a cat with a pearl earring,",negative_prompt="obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands, extra fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, low quality, long neck, frame, text, big, 3d rendering, unreal",control_image=depth_map,strength=0.8)

save_path = "./tmp_ip_adpater_result"
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i, it in enumerate(images):
    save_name = os.path.join(save_path, "%d.jpg" % i)
    it.save(save_name)
