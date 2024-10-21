from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, \
    DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import safetensors.torch as sf
import math
import PIL.Image as Image
import numpy as np
import cv2

import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from plugin_modules.DesignEdit.process_utils import pytorch2numpy, numpy2pytorch, resize_and_center_crop, \
    resize_without_crop

# init model
sd15_name = './models/ic_lighting/stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")

# Change UNet
with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride,
                                  unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

import pdb

pdb.set_trace()
# Load
model_path = './models/ic_lighting/iclight_sd15_fc.safetensors'
sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}

# sd_merged = {}
# for k in sd_origin.keys():
#     if k.find("conv_in") != -1:
#         sd_merged[k] = sd_origin[k] + sd_offset[k]
#     else:
#         sd_merged[k] = sd_origin[k]
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys
# 每个sd_merged对应的key，在原来的unet中基本都有，相当于完全覆盖了
# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers
dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


# param setting
fg_img_name = "./tmp_example_img/lighting/fg_cat.png"
bg_img_name = "./tmp_example_img/lighting/bg_indoor.png"
prompt = "sunshine from window"
image_width = 1024
image_height = 1024
seed = 12345
steps = 25
num_samples = 1
cfg = 2
lowres_denoise = 0.9
highres_scale = 1.5
highres_denoise = 0.5
a_prompt = "best quality"
n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

fg_img = cv2.imread(fg_img_name)
bg_img = cv2.imread(bg_img_name)
# left light
# gradient = np.linspace(255, 0, image_width)
# image = np.tile(gradient, (image_height, 1))
# input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
input_bg = bg_img

rng = torch.Generator(device=device).manual_seed(int(seed))
fg = resize_and_center_crop(fg_img, image_width, image_height)

concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)  # 1,3,1024,1024
concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor  # 1,4,128,128

conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)
# conds, 1,77,768
# unconds, 1,77,768
if input_bg is None:
    latents = t2i_pipe(
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor
else:
    bg = resize_and_center_crop(input_bg, image_width, image_height)  # 1024,1024,3
    bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)  # 1,3,1024,1024
    bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor  # 1,4,128,128
    latents = i2i_pipe(
        image=bg_latent,
        strength=lowres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / lowres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

import pdb

pdb.set_trace()
pixels = vae.decode(latents).sample
pixels = pytorch2numpy(pixels)
pixels = [resize_without_crop(
    image=p,
    target_width=int(round(image_width * highres_scale / 64.0) * 64),
    target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
latents = latents.to(device=unet.device, dtype=unet.dtype)

image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

fg = resize_and_center_crop(fg_img, image_width, image_height)
concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

latents = i2i_pipe(
    image=latents,
    strength=highres_denoise,
    prompt_embeds=conds,
    negative_prompt_embeds=unconds,
    width=image_width,
    height=image_height,
    num_inference_steps=int(round(steps / highres_denoise)),
    num_images_per_prompt=num_samples,
    generator=rng,
    output_type='latent',
    guidance_scale=cfg,
    cross_attention_kwargs={'concat_conds': concat_conds},
).images.to(vae.dtype) / vae.config.scaling_factor

pixels = vae.decode(latents).sample

result = pytorch2numpy(pixels)

a = 0
