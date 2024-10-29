import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))
from plugin_modules.ip_adapter.ip_adapter import IPAdapterXL, IPAdapterPlusXL
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline, \
    StableDiffusionXLControlNetPipeline, ControlNetModel,DPMSolverMultistepScheduler
# control inpainting, control img2img,
import torch
from PIL import Image
import cv2
import numpy as np
import argparse
import json
import random

def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    # general param
    parser.add_argument('--seed', default=53, type=int)
    # save path
    parser.add_argument('--save_dir', default='results/base_infer_result', type=str)

    # json path
    parser.add_argument('--json_file', default='./flux_json/animal.json')
    # base model path
    parser.add_argument('--flux_model', default='./models/wildcardxXL', type=str)

    # pos type
    parser.add_argument('--style', default='animal', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # make base folder
    save_dir = args.save_dir
    benchmark_file_path = args.json_file
    os.makedirs(save_dir, exist_ok=True)

    # base param
    base_seed = args.seed
    MAX_SEED = np.iinfo(np.int32).max

    torch_dtype = torch.float16
    model_type = "fp16"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # inference param
    num_inference_steps = 20
    max_gen_num = 4
    guidance_scale = 5.0
    # pos_prefix = "masterpiece, watercolor style, close-up front portrait photo"
    # pos_suffix = "realistic, photographic, best-quality, High Resolution, intricate detail"

    neg_prefix = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, low quality, long neck, frame, text, worst quality,watermark, deformed, ugly,  blur, out of focus,extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, "
    image_encoder_path = "./models/IP-Adapter/sdxl_model/image_encoder"
    ip_ckpt = "./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors"

    # load flux model
    pipe = StableDiffusionXLPipeline.from_pretrained(args.flux_model,
                                                     torch_dtype=torch_dtype,
                                                     # torch_dtype=torch.bfloat16,
                                                     use_safetensors=True,
                                                     variant=model_type
                                                     ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=4)  # modify the num_tokens=16

    # animal
    if args.style == "animal":
        pos_prefix = "masterpiece, debluring, frontal face"
        pos_suffix = "Smooth strokes, Pastel colors, Low saturation, Create artwork, masterpiece,best-quality, High Resolution, intricate detail"

    if args.style == "animal_gaze":
        pos_prefix = "masterpiece, debluring, close-up portrait photo, Detailed clear face"
        pos_suffix = "Smooth strokes, Pastel colors, Low saturation, Create artwork, masterpiece,best-quality, High Resolution, intricate detail"

    # stamp gaze
    if args.style == "stamp_gaze":
        pos_prefix = "masterpiece,hyperrealism, close-up portrait photo, Detailed clear face"
        pos_suffix = "hyperrealism, masterpiece,best-quality, High Resolution, intricate detail"

    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
        benchmark_json = json.load(f)

    for key in benchmark_json.keys():
        if key not in ['Manul']:
            continue
        print("start to gen ", '-' * 30, key)
        prompt = benchmark_json[key]['prompt']
        # negative_prompt = benchmark_json[key]['negative_prompt']
        image_path = benchmark_json[key]['image_path']

        ref_image = Image.open(image_path)

        print(prompt)
        input_prompt = pos_prefix + ',' + prompt + ',' + pos_suffix
        input_negative_prompt = neg_prefix #+ ',' + negative_prompt

        base_seed = benchmark_json[key].get('seed', random.randint(0, MAX_SEED))

        images = ip_model.generate(pil_image=ref_image, num_samples=max_gen_num, num_inference_steps=20,
                                   prompt=prompt, scale=1.0, negative_prompt=input_negative_prompt)

        for gen_time in range(max_gen_num):
            save_image_path = os.path.join(save_dir, f'{key}_seed_{base_seed}_cnt_{gen_time}_result.png')
            images[gen_time].save(save_image_path)
