import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler

import cv2
import torch
import numpy as np
from PIL import Image

import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from insightface.app import FaceAnalysis
from plugin_modules.instantID.pipeline_stable_diffusion_xl_instantid_full_style import \
    StableDiffusionXLInstantIDPipeline, \
    draw_kps

# define params
enhance_face_region = False#True
adapter_strength_ratio = 0.8
style_scale = 1.2
guidance_scale = 5
enable_LCM = False
num_steps = 5 if enable_LCM else 30
identitynet_strength_ratio = 0.5


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# prepare 'antelopev2' under ./models
app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# prepare models under ./checkpoints
face_adapter = f'./models/instantID_model/ip-adapter.bin'
controlnet_path = f'./models/instantID_model/ControlNetModel'

# style ip-adapter
image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"

# save path
save_path = "./tmp_diffuser_plugin/instant_style_result/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load IdentityNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

# base_model = 'wangqixun/YamerMIX_v8'  # from https://civitai.com/models/84040?modelVersionId=196039
base_model = "./models/instantID_model/YamerMIX_v8"
base_model_file = "/data/tc_guo/models/sfmodels/samaritan3dCartoon_v40SDXL.safetensors"
base_model_file = "/data/tc_guo/models/sfmodels/animagine-xl-3.1.safetensors"
base_model_file = "/data/tc_guo/models/sfmodels/dreamshaperXL10_alpha2Xl10.safetensors"
pipe = StableDiffusionXLInstantIDPipeline.from_single_file(
    base_model_file,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.cuda()
pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.set_ip_adapter_scale(id_scale=adapter_strength_ratio, style_scale=style_scale)

# load adapter
pipe.load_ip_adapter_instantid(model_ckpt=face_adapter, ip_adapter_ckpt=ip_ckpt, image_encoder_path=image_encoder_path,
                               target_blocks=["up_blocks"], id_scale=adapter_strength_ratio,
                               style_scale=style_scale)  # up_blocks.0.attentions.1

# load and disable LCM
# pipe.load_lora_weights("./models/instantID_model/latent-consistency/lcm-lora-sdxl")
# if enable_LCM:
#     pipe.enable_lora()
#     pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# else:
#     pipe.disable_lora()
# lora_path = "/data/tc_guo/models/sfmodels/cartoon_stickers_xl_v1.safetensors"
# pipe.load_lora_weights(lora_path)
# pipe.enable_lora()

# load an image
# face_image = load_image("./tmp_example_img/face_img.jpg")
face_image = load_image("./tmp_example_img/human_img/liuyifei.png")
face_image = resize_img(face_image)
face_image_cv2 = convert_from_image_to_cv2(face_image)
height, width, _ = face_image_cv2.shape

# prepare face emb
face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
if len(face_info) == 0:
    raise ("Cannot find any face in the image! Please upload another person image")

face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
    -1]  # only use the maximum face
face_emb = face_info['embedding']
face_kps = draw_kps(face_image, face_info['kps'])  # different from the app

# load pose image
pose_image = load_image("./tmp_example_img/human_img/ref_image2.png")
pose_image = resize_img(pose_image)
pose_image_cv2 = convert_from_image_to_cv2(pose_image)

face_info = app.get(pose_image_cv2)
if len(face_info) == 0:
    raise ("Cannot find any face in the image! Please upload another person image")
face_info = face_info[-1]
face_kps = draw_kps(pose_image, face_info['kps'])

width, height = face_kps.size

if enhance_face_region:
    control_mask = np.zeros([height, width, 3])
    x1, y1, x2, y2 = face_info["bbox"]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    control_mask[y1:y2, x1:x2] = 255
    control_mask = Image.fromarray(control_mask.astype(np.uint8))
else:
    control_mask = None

# style image
style_img_name = "./tmp_example_img/instant_style/cartoonlady.jpeg"
style_img = Image.open(style_img_name)
style_img = resize_img(style_img)
neg_content_prompt = "a lady portrait"  # a lady portrait
neg_content_scale = 0.5

prompt = 'a photo of a lady'  # in Mona Lisa style'

negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, look awry"

generator = torch.Generator(device="cuda").manual_seed(1024)

# generate image
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image_embeds=face_emb,
    image=face_kps,
    control_mask=control_mask,
    controlnet_conditioning_scale=float(identitynet_strength_ratio),
    num_inference_steps=num_steps,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    # style
    style_image=style_img,
    neg_content_prompt=neg_content_prompt,
    neg_content_scale=neg_content_scale,
    # generator=generator,
    # ip_adapter_scale=0.8,
).images[0]

image.save(os.path.join(save_path, 'id_style_result.jpg'))
