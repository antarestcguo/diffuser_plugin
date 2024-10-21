from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from random import choice
from pathlib import Path
from transformers import CLIPImageProcessor
import numpy as np
import os
import random
import torch

import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from plugin_modules.ip_adapter.gen_basewordimg_v2_clean import gen_character, is_chinese
from plugin_modules.ip_adapter.word_art_config_dict import style_config_dict

CN_font_file_dict = {
    'baoli': {"file_path": './resource/font/STBaoliSC-Regular-01.ttf', "font_size": 812},
    'QingNiaoxingkai': {"file_path": "./resource/font/Qingniao_modify.ttf", "font_size": 812, }
}
EN_font_file_dict = {
    "HFPuff": {"file_path": "./resource/font/HFPuff-2.ttf", "font_size": 812},
}
EN_default_font_path = ['./resource/font/ArialNarrowBold.ttf']
CN_default_font_path = ['./resource/font/STBaoliSC-Regular-01.ttf', './resource/font/ArialNarrowBold.ttf']

style_path = "./tmp_example_img/style_imgs/"

# set hardly
i_drop_rate = 0.05
t_drop_rate = 0.05
ti_drop_rate = 0.05


class X2PaintingControlDataset(Dataset):
    def __init__(
            self,
            image_dir,  # xx
            image_file_name,  # xx/维_winter_black/xxx.jpg
            encoders,
            tokenizers,
            size=1024,
    ):
        self.image_dir = image_dir
        self.image_name_list = []

        # prompt: call style_config_dict

        with open(image_file_name, 'r') as f:
            for line in f.readlines():
                self.image_name_list.append(line.strip())

        self.encoders = encoders
        self.tokenizers = tokenizers
        self.size = size
        self._length = len(self.image_name_list)

        self.fg_color = (0, 0, 0)
        self.bg_color = (255, 255, 255)

        self.clip_image_processor = CLIPImageProcessor()

    def __len__(self):
        return self._length

    def tokenize_prompt(self, tokenizer, prompt):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(self, text_encoders, tokenizers, prompt, text_input_ids_list=None):
        prompt_embeds_list = []

        for i, text_encoder in enumerate(text_encoders):
            if tokenizers is not None:
                tokenizer = tokenizers[i]
                text_input_ids = self.tokenize_prompt(tokenizer, prompt)
            else:
                assert text_input_ids_list is not None
                text_input_ids = text_input_ids_list[i]

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def compute_text_embeddings(self, prompt, text_encoders, tokenizers):
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(text_encoders, tokenizers, prompt)
        return prompt_embeds, pooled_prompt_embeds

    def preprocess(self, image):  # input
        # resize
        image = image.resize((self.size, self.size))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, index):
        img_file_name = self.image_name_list[index]
        # split
        folder = img_file_name.split("/")[-2]
        tokens = folder.split('_')
        word = tokens[0]
        style_name = "_".join(tokens[1:])

        # read image
        img = Image.open(os.path.join(self.image_dir, img_file_name))
        img = self.preprocess(img)

        # get font param
        char_type = "CN" if is_chinese(word) else "EN"
        default_font_path_list = eval(char_type + "_default_font_path")
        font_style = choice(style_config_dict[style_name][(char_type + "_font_style").replace("CN_", "")])
        font_path = CN_font_file_dict[font_style]['file_path']

        # gen word image: PIL.Image
        base_img, _, _, _, _ = gen_character(word,
                                             font_path,
                                             default_font_path_list,
                                             self.fg_color, self.bg_color, char_type, max_side=self.size,
                                             min_side=self.size)
        base_img = self.preprocess(base_img)

        # read ipadapter image
        style_folder = os.path.join(style_path, style_name)
        style_img_list = os.listdir(style_folder)
        style_img_name = os.path.join(style_folder, style_img_list[0])
        style_img = Image.open(style_img_name)
        style_img = self.clip_image_processor(images=style_img, return_tensors="pt").pixel_values

        # encode prompt
        prompt = style_config_dict[style_name]["prompt"]
        rand_num = random.random()
        drop_image_embed = 0
        if rand_num < i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (i_drop_rate + t_drop_rate):
            prompt = ""
        elif rand_num < (i_drop_rate + t_drop_rate + ti_drop_rate):
            prompt = ""
            drop_image_embed = 1

        instance_prompt_hidden_states, instance_pooled_prompt_embeds = self.compute_text_embeddings(
            prompt, self.encoders, self.tokenizers
        )

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (self.size, self.size)
        target_size = (self.size, self.size)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        batch_num = 1
        add_time_ids = add_time_ids.repeat(batch_num, 1)

        example = {}
        example["control_images"] = torch.from_numpy(base_img).permute(2, 0, 1)
        example["instance_images"] = torch.from_numpy(img).permute(2, 0, 1)
        example["adapter_images"] = style_img

        example['prompt_embeds'] = instance_prompt_hidden_states
        example['unet_add_text_embeds'] = instance_pooled_prompt_embeds

        # for sdxl
        example['time_ids'] = add_time_ids

        # class free training
        example["drop_image_embed"] = drop_image_embed

        return example
