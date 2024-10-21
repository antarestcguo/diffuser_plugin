from controlnet_aux import LineartDetector
from controlnet_aux import HEDdetector
import os
import PIL.Image as Image

processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
# hed = HEDdetector.from_pretrained('/data/tc_guo/models/SDXL-CONTROLNET-DOWNLOAD/hed')

input_img_file = "/home/Antares.Guo/code/ToonCrafter-main/prompts/fire_part2/fire_frame3.jpeg"
image = Image.open(input_img_file)
control_image = processor(image)
# control_image = hed(image, scribble=True)
save_path = "/home/Antares.Guo/code/ToonCrafter-main/prompts/fire_edge"

control_image.save(os.path.join(save_path, "lineart_control_fire_frame3.jpg"))
