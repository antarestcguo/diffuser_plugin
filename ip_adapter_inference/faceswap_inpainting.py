import os
import sys

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from PIL import Image
from plugin_modules.ip_adapter import IPAdapter,IPAdapterXL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, \
    DDIMScheduler, AutoencoderKL,StableDiffusionInpaintPipeline

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline

device = "cuda"

# SD1.5
# base_model_path = "./models/stable-diffusion-v1-5"
# vae_model_path = "./models/IP-Adapter/stabilityai/sd-vae-ft-mse"
# image_encoder_path = "./models/IP-Adapter/models/image_encoder/"
# ip_ckpt = "./models/IP-Adapter/models/ip-adapter_sd15.bin"
# noise_scheduler = DDIMScheduler(
#     num_train_timesteps=1000,
#     beta_start=0.00085,
#     beta_end=0.012,
#     beta_schedule="scaled_linear",
#     clip_sample=False,
#     set_alpha_to_one=False,
#     steps_offset=1,
# )
# vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
# pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     scheduler=noise_scheduler,
#     vae=vae,
#     feature_extractor=None,
#     safety_checker=None
# )
# ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

# SDXL
# base_model_path = "./models/stable-diffusion-xl-base-1.0"
# # vae_model_path = "./models/IP-Adapter/stabilityai/sd-vae-ft-mse"
# image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
# ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"
# pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     add_watermarker=False,
# )
# # load ip-adapter
# ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# SDXL FaceIDXL
base_model_path = "./models/stable-diffusion-xl-base-1.0"
ip_ckpt = "./models/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin"
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    add_watermarker=False,
)
# load ip-adapter
ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

masked_image_name = "./tmp_example_img/inpainting/image.png"
mask_name = "./tmp_example_img/inpainting/mask.png"
ref_name = "./tmp_example_img/images/girl.png"

save_path = "./tmp_faceswap_inpainting_result"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# read image prompt
image = Image.open(ref_name)
image.resize((256, 256))
masked_image = Image.open(masked_image_name).resize((512, 768))
mask = Image.open(mask_name).resize((512, 768))

# FaceID
import cv2
from insightface.app import FaceAnalysis
import torch

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
image = cv2.imread(ref_name)
faces = app.get(image)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50,
                           # seed=42,
                           image=masked_image, mask_image=mask, strength=0.7, )
for i, it in enumerate(images):
    save_name = os.path.join(save_path, "example_SDXL_%i.jpg" % i)
    it.save(save_name)
