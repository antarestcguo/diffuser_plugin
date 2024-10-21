import cv2


filename_list = ['cartoon_fire_sample0.mp4',
                 'cartoon_fire_sample1.mp4',
                 'cartoon_fire_sample2.mp4',
                 'cartoon_fire_sample3.mp4']
# default w1=w2

cap1 = cv2.VideoCapture(filename_list[0])
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap1.get(cv2.CAP_PROP_FPS))
cap1.release()

outfile_name = './cartoon_fire_sample_total.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(outfile_name,fourcc,10,(width,height),True)

cnt = 0
for it_file in filename_list:
    cap = cv2.VideoCapture(it_file)
    isOpened = cap.isOpened
    while isOpened:
        ret,frame = cap.read()

        if not ret:
            break

        output.write(frame)
        print("cnt", cnt)
        cnt += 1
    cap.release()

output.release()


