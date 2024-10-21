import sys
import os
sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))
from plugin_modules.ip_adapter.ip_adapter import IPAdapterXL,IPAdapterPlusXL
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline
# control inpainting, control img2img,
import torch
from PIL import Image

negative_prompt="obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, low quality, long neck, frame, text, worst quality,watermark, deformed, ugly,  blur, out of focus,extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, "

device = "cuda"
save_path = "./tmp_diffuser_plugin/tmp_c_result"
if not os.path.exists(save_path):
    os.makedirs(save_path)
# define image
face_image = "./tmp_resource/ipadapter_resource/snowleopard.jpeg"
ref_image = ""

# base model
base_model_file = "/data/tc_guo/models/sfmodels/samaritan3dCartoon_v40SDXL.safetensors"
base_model_file = "/data/tc_guo/models/sfmodels/dreamshaperXL10_alpha2Xl10.safetensors"
# base_model_file = "/data/tc_guo/models/sfmodels/Juggernaut.safetensors"
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
# xlplus
image_encoder_refine_path = "./models/IP-Adapter/models/image_encoder"
ip_ckpt_refine = "./models/IP-Adapter/sdxl_model/ip-adapter-plus_sdxl_vit-h.safetensors"
pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
    base_model_file,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
lora_path = ""
# pipe.load_lora_weights(lora_path)
# reduce GPU MEM
pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()
ip_model = IPAdapterPlusXL(pipe, image_encoder_refine_path, ip_ckpt_refine, device,num_tokens=16) # modify the num_tokens=16

image = Image.open(face_image)
# generate
images = ip_model.generate(pil_image=image, image=image,num_samples=4, num_inference_steps=20,
        prompt="a snow leopard, wearing large glasses, Metal eyeglass frame", scale=1.0,strength=0.7,
                           negative_prompt=negative_prompt)

# save
for i,it in enumerate(images):
    save_name = os.path.join(save_path,"{}_plus_scale1.0_snowleopard.jpg".format(i))
    it.save(save_name)





