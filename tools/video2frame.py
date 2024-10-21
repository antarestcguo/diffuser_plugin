import cv2
import os

video_name = "./tmp_example_img/ipadater_img/wolf/wolf.mp4"
save_path = "./tmp_example_img/ipadater_img/wolf/"

cap = cv2.VideoCapture(video_name)
fps = cap.get(cv2.CAP_PROP_FPS)

isOpened = cap.isOpened  # 判断是否打开
i = 0
save_interval = 10
while (isOpened):
    (flag, frame) = cap.read()  # 读取每一帧，一张图像flag 表明是否读取成果 frame内容
    if flag == True and i % save_interval == 0:
        fileName = os.path.join(save_path, "frame_%s.jpg" % i)
        print(fileName)
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    i += 1
