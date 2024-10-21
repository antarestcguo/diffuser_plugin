import sys
import os
import PIL.Image as Image

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../../"))

from plugin_modules.ip_adapter.gen_baseword import paste_maxtext

save_path = "./tmp_gen_enhance_text_image"
if not os.path.exists(save_path):
    os.makedirs(save_path)

font_file_dict = {
    'baoli': {"file_path": './resource/font/STBaoliSC-Regular-01.ttf', "font_size": 812},
    # 'Arial': './resource/font/ArialNarrowBold.ttf',
    # 'tetai': {"file_path": './resource/font/tetai-2.ttf', "font_size": 595},

    # 'RuiZiYunxingkai': {"file_path": "./resource/font/RuiZiYunZiKuXingKaiGB-2.ttf", "font_size": 812, },
    # 'HanYixingkai': {"file_path": "./resource/font/HanYiXingKaiJian-1.ttf", "font_size": 812, },
    # 'JiZiJingDianxingkai': {
    #     "file_path": "./resource/font/JiZiJingDianXingKaiJianFan-Shan(GEETYPE-XingKaiGBT-Flash)-2.ttf",
    #     "font_size": 812, },
    'QingNiaoxingkai': {"file_path": "./resource/font/QingNiaoHuaGuangXingKai-2.ttf", "font_size": 812, }
}
word_dict = {
    # "wang": "王",
    # "li": "李",
    # "zhang": "张",
    # "liu": "刘",
    # "chen": "陈",
    # "yang": "杨",
    # "zhao": "赵",
    # "huang": "黄",
    # "zhou": "周",
    # "wu": "吴",
    "liang": "梁",
    "xie": "谢",
    "dong": "董",
    "wei": "魏",
    "dai": "戴",
}
bg_color = (250, 248, 235)
fg_color = (98, 104, 114)

input_image_name = "./tmp_ip_adpater_result_select_max/shuimowood2_black_song/wei_1_strength_0.800.jpg"

ori_image = Image.open(input_image_name)

enhance_image, resize_image_text = paste_maxtext(word_dict['wei'], ori_image, fg_color, bg_color,
                                                 font_file_dict['QingNiaoxingkai']['file_path'], font_size=812,
                                                 alpha_ratio=0.85, b_bg=True)

save_name = os.path.join(save_path, "wei_enhance.jpg")
enhance_image.save(save_name)
save_name = os.path.join(save_path, "wei_text.jpg")
resize_image_text.save(save_name)
