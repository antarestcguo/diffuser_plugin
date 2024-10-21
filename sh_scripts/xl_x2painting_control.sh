#CUDA_VISIBLE_DEVICES=0 python x2painting_train_scripts/train_control_sdxl.py \
#--pretrained_model_name_or_path ./models/stable-diffusion-xl-base-1.0 \
#--controlnet_model_name_or_path ./models/QRcodeCN \
#--output_dir /data/tc_guo/train_model/x2painting_CN/init \
#--image_dir /data/tc_guo/infer_result/tmp_diffuser_plugin \
#--image_file_name /data/tc_guo/infer_result/tmp_diffuser_plugin/x2painting_baseimg.txt \
#--image_encoder_path ./models/IP-Adapter/sdxl_model/image_encoder \
#--pretrained_ip_adapter_path ./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors \
#--learning_rate 5e-6 \
#--num_train_epochs 1

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
--main_process_port 20010 x2painting_train_scripts/train_control_sdxl.py \
--pretrained_model_name_or_path ./models/stable-diffusion-xl-base-1.0 \
--controlnet_model_name_or_path ./models/QRcodeCN \
--output_dir /data/tc_guo/train_model/x2painting_CN/init \
--image_dir /data/tc_guo/infer_result/tmp_diffuser_plugin \
--image_file_name /data/tc_guo/infer_result/tmp_diffuser_plugin/x2painting_baseimg.txt \
--image_encoder_path ./models/IP-Adapter/sdxl_model/image_encoder \
--pretrained_ip_adapter_path ./models/IP-Adapter/sdxl_model/ip-adapter_sdxl.safetensors \
--learning_rate 5e-6 \
--num_train_epochs 2

