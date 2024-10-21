import os

# xx/word_style_x/xx.jpg
root_path = "/data/tc_guo/infer_result/tmp_diffuser_plugin"
sub_folder = "tmp_ip_adpater_qrcode_control_4shape"
save_file_name = '/data/tc_guo/infer_result/tmp_diffuser_plugin/x2painting_basecontrol.txt'

work_path = os.path.join(root_path, sub_folder)
folder_list = os.listdir(work_path)

with open(save_file_name, 'w') as f:
    for it_folder in folder_list:
        image_work_path = os.path.join(work_path, it_folder)
        image_list = os.listdir(image_work_path)

        for it_image_name in image_list:
            if it_image_name.find(".jpg") == -1:
                continue
            image_name_str = os.path.join(sub_folder, it_folder, it_image_name)

            f.write(image_name_str + '\n')
