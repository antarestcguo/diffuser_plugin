import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler

import cv2
import torch
import numpy as np
from PIL import Image

# import sys
# sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from insightface.app import FaceAnalysis
from plugin_modules.instantID.pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline, \
    draw_kps

# define params
enhance_face_region = True
adapter_strength_ratio = 0.8
guidance_scale = 5
enable_LCM = False
num_steps = 5 if enable_LCM else 30
identitynet_strength_ratio = 0.8


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

# load IdentityNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

# base_model = 'wangqixun/YamerMIX_v8'  # from https://civitai.com/models/84040?modelVersionId=196039
base_model = "./models/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.cuda()
pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.set_ip_adapter_scale(adapter_strength_ratio)

# load adapter
pipe.load_ip_adapter_instantid(face_adapter)

# load and disable LCM
pipe.load_lora_weights("./models/instantID_model/latent-consistency/lcm-lora-sdxl")
if enable_LCM:
    pipe.enable_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
else:
    pipe.disable_lora()

# load an image
# face_image = load_image("./tmp_example_img/face_img.jpg")
face_image = load_image("./tmp_example_img/guodegang_face.jpeg")
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
pose_image = load_image("./tmp_example_img/ref_image.png")
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

# prompt
# prompt = "film noir style, ink sketch|vector, highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
# negative_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"

prompt = 'waist-up "a photo of a person in a Jungle"  by Syd Mead, tangerine cold color palette, muted colors, detailed, 8k,photo r3al,dripping paint,3d toon style,3d style,Movie Still'

negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green"

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
    # generator=generator,
    # ip_adapter_scale=0.8,
).images[0]

image.save('./result.jpg')
