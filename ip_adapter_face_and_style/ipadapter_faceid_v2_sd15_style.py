from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL, StableDiffusionPipeline
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../"))

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from plugin_modules.instantstyle.ipadapter.ip_adapter_faceid_instantstyle import IPAdapterFaceIDXL, \
    IPAdapterFaceIDPlusXL, IPAdapterFaceIDPlus
import torch
import cv2

save_path = "./tmp_diffuser_plugin/ipadapter_faceidplusv2_result_sd15/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# initialization
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "/data/tc_guo/models/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sd15.bin"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

# load SDXL pipeline
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
).to(device)

pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()
# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# app2 = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app2.prepare(ctx_id=0, det_size=(640, 640))

face_image_file = ["./tmp_example_img/human_img/liuyifei.png"]
prompt = "a photo of lady in Da Vinci style, (masterpiece:1.2),(best quality:1.2),"
negative_prompt = "low quality, (naked:1.2), bikini, skimpy, (scanty:1.5), (bare skin:1.2), lingerie, swimsuit, (exposed:1.2), see-through"  # (lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry,

face_strength = 1.3
likeness_strength = 1.0

faceid_all_embeds = []
first_iteration = True
preserve_face_structure = True
for image in face_image_file:
    face = cv2.imread(image)
    faces = app.get(face)
    faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    # faceid_embed = torch.from_numpy(faces[0].embedding).unsqueeze(0)
    # import pdb
    #
    # pdb.set_trace()
    # import numpy as np
    #
    # face_info = app2.get(cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR))
    # face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
    #     -1]  # only use the maximum face
    # face_emb = face_info['embedding']
    # face_kps = face_info['kps']
    # faceid_embed = torch.from_numpy(face_info.normed_embedding).unsqueeze(0)

    faceid_all_embeds.append(faceid_embed)
    if (first_iteration and preserve_face_structure):
        face_image = face_align.norm_crop(face, landmark=faces[0].kps, image_size=224)  # you can also segment the face
        first_iteration = False

average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)

images = ip_model.generate(pil_image=style_img,
                           prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=average_embedding,
                           scale=likeness_strength, face_image=face_image, shortcut=True, num_samples=4,
                           s_scale=face_strength, width=1024, height=1024, num_inference_steps=30, guidance_scale=7.5,
                           seed=None,

                           )
print(negative_prompt)
for i, it in enumerate(images):
    save_name = os.path.join(save_path,
                             "%d.jpg" % (i))
    it.save(save_name)
