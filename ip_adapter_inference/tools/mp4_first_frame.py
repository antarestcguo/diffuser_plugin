import cv2
import os

video_path = "../../tmp_video"
save_path = "../../tmp_video_frame"
if not os.path.exists(save_path):
    os.makedirs(save_path)
video_list = os.listdir(video_path)

for it_video in video_list:
    video_name = os.path.join(video_path, it_video)
    cap = cv2.VideoCapture(video_name)
    while cap.isOpened():
        ret, frame = cap.read()
        s_n, s_e = os.path.splitext(it_video)
        file_name = os.path.join(save_path, s_n + '.jpg')
        cv2.imwrite(file_name, frame)
        break
