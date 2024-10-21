import cv2

image_file = "./tmp_example_img/instant_style/cartoonlady.jpeg"

img = cv2.imread(image_file)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

import pdb

pdb.set_trace()

a = 0
