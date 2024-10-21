import json
import os
import cv2
import numpy as np

img_dir = "./tmp_example_img/watermark"
json_dir = "./tmp_example_img/watermark"
mask_dir = "./tmp_example_img/watermark"

if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

json_list = os.listdir(json_dir)

for it_json in json_list:
    if it_json.find(".json") == -1:
        continue

    with open(os.path.join(json_dir, it_json), 'r') as f:
        data = f.read()

    json_data = json.loads(data)
    imageHeight = json_data["imageHeight"]
    imageWidth = json_data["imageWidth"]

    mask_img = np.zeros([imageHeight, imageWidth, 3], dtype=np.uint8)

    mask_name = os.path.join(mask_dir, it_json[:-5] + '_mask.jpg')
    # get the points
    for it_label in json_data["shapes"]:
        points = it_label["points"]
        points = np.array(points, dtype=np.int32)  # tips: points location must be int32

        # fill the contour with 255
        cv2.fillPoly(mask_img, [points], (255, 255, 255))

    # save the mask
    cv2.imwrite(mask_name, mask_img)
