import os
from diffusers import StableDiffusionXLPipeline, FluxPipeline, FluxImg2ImgPipeline
import cv2
import argparse
import torch
import json
import numpy as np
import random
from PIL import Image
import sys
sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))
from plugin_modules.instantstyle.ipadapter.image_process import resize_img


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    # general param
    parser.add_argument('--seed', default=53, type=int)
    # save path
    parser.add_argument('--save_dir', default='results/base_infer_result', type=str)

    # json path
    parser.add_argument('--json_file', default='./flux_josn/animal.json')
    # base model path
    parser.add_argument('--flux_model', default='./models/Flux', type=str)

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
    num_inference_steps = 50
    max_gen_num = 4
    guidance_scale = 0.0
    # pos_prefix = "masterpiece, watercolor painting, ink style, painterly, detailed, textural, artistic"
    pos_prefix = "masterpiece, debluring, close-up front portrait photo, Detailed clear face, frontal face"
    pos_suffix = "realistic, photographic, best-quality, High Resolution, intricate detail"

    # load flux model
    pipe = FluxImg2ImgPipeline.from_pretrained(args.flux_model,
                                        torch_dtype=torch.bfloat16,
                                        use_safetensors=True).to(device)
    pipe.enable_model_cpu_offload()

    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
        benchmark_json = json.load(f)

    for key in benchmark_json.keys():
        print("start to gen ", '-' * 30, key)
        prompt = benchmark_json[key]['prompt']
        image_path = benchmark_json[key]['image_path']
        strength = benchmark_json[key]['strength']

        if not os.path.exists(image_path):
            continue
        image = Image.open(image_path)
        white_img = Image.new("RGB",image.size,(255,255,255))
        white_img.paste(image,(0,0,white_img.size[0],white_img.size[1]))
        print(prompt)
        input_prompt = pos_prefix + ',' + prompt + ',' + pos_suffix
        resize_input = resize_img(white_img)
        import pdb
        pdb.set_trace()
        for gen_time in range(max_gen_num):
            base_seed = random.randint(0, MAX_SEED)
            images = pipe(
                prompt=input_prompt,
                image=resize_input,
                strength=strength,
                num_inference_steps=num_inference_steps,
                # num_images_per_prompt=max_gen_num,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device).manual_seed(base_seed),
            ).images

            save_image_path = os.path.join(save_dir, f'{key}_seed_{base_seed}_cnt_{gen_time}_result.png')
            images[0].resize(resize_input.size).save(save_image_path)
