import os
from diffusers import StableDiffusionXLPipeline, FluxPipeline,DPMSolverMultistepScheduler,FlowMatchEulerDiscreteScheduler
import cv2
import argparse
import torch
import json
import numpy as np
import random


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
    num_inference_steps = 20
    max_gen_num = 4
    guidance_scale = 3.5
    # pos_prefix = "masterpiece, watercolor style, close-up front portrait photo"
    pos_prefix = "masterpiece, debluring, close-up front portrait photo, Detailed clear face, frontal face"
    # pos_suffix = "realistic, photographic, best-quality, High Resolution, intricate detail"
    pos_suffix = "Smooth strokes, Pastel colors, Low saturation, Create artwork, masterpiece,best-quality, High Resolution, intricate detail"
    # load flux model
    pipe = FluxPipeline.from_pretrained(args.flux_model,
                                        torch_dtype=torch_dtype,
                                        use_safetensors=True,
                                        variant=model_type).to(device)
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(os.path.join(args.flux_model,'scheduler_flux/scheduler_config.json'),timestep_spacing="trailing")
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    #     os.path.join(args.flux_model, 'scheduler_dpm/scheduler_config.json'))
    pipe.enable_model_cpu_offload()

    with open(benchmark_file_path, 'r', encoding='utf-8') as f:
        benchmark_json = json.load(f)

    for key in benchmark_json.keys():
        print("start to gen ", '-' * 30, key)
        prompt = benchmark_json[key]['prompt']
        print(prompt)
        input_prompt = pos_prefix + ',' + prompt + ',' + pos_suffix

        base_seed = benchmark_json[key].get('seed', random.randint(0, MAX_SEED))
        images = pipe(
            prompt=input_prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=max_gen_num,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            generator=torch.Generator(device).manual_seed(base_seed),
        ).images
        for gen_time in range(max_gen_num):
            save_image_path = os.path.join(save_dir, f'{key}_seed_{base_seed}_cnt_{gen_time}_result.png')
            images[gen_time].save(save_image_path)
