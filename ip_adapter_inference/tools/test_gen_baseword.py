import sys
import os

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../../"))

from plugin_modules.ip_adapter.gen_basewordimg_v2 import gen_character, is_chinese, gen_word_EN, gen_word_CN_together

save_path = "./tmp_gen_basewordmax"
if not os.path.exists(save_path):
    os.makedirs(save_path)

CN_font_file_dict = {
    # 'baoli': {"file_path": './resource/font/STBaoliSC-Regular-01.ttf', "font_size": 812},
    # 'Arial': './resource/font/ArialNarrowBold.ttf',
    # 'tetai': {"file_path": './resource/font/tetai-2.ttf', "font_size": 595},

    # 'RuiZiYunxingkai': {"file_path": "./resource/font/RuiZiYunZiKuXingKaiGB-2.ttf", "font_size": 812, },
    # 'HanYixingkai': {"file_path": "./resource/font/HanYiXingKaiJian-1.ttf", "font_size": 812, },
    # 'JiZiJingDianxingkai': {
    #     "file_path": "./resource/font/JiZiJingDianXingKaiJianFan-Shan(GEETYPE-XingKaiGBT-Flash)-2.ttf",
    #     "font_size": 812, },
    'QingNiaoxingkai': {"file_path": "./resource/font/QingNiaoHuaGuangXingKai-2.ttf", "font_size": 812, }
}
EN_font_file_dict = {
    # "Monster": {"file_path": "./resource/font/A-Little-Monster-2.ttf", "font_size": 812},
    "HFPuff": {"file_path": "./resource/font/HFPuff-2.ttf", "font_size": 812},
    # "HFShinySunday": {"file_path": "./resource/font/HFShinySunday-2.ttf", "font_size": 812},

    # "HFSoda": {"file_path": "./resource/font/HFSoda-2.ttf", "font_size": 812},
    # "justanotherhand": {"file_path": "./resource/font/justanotherhand-regular-2.ttf", "font_size": 812},
}
EN_default_font_path = ['./resource/font/ArialNarrowBold.ttf']
CN_default_font_path = ['./resource/font/STBaoliSC-Regular-01.ttf', './resource/font/ArialNarrowBold.ttf']

# save_name: chinese_word
# word_dict = {
#     # "lang": "朤",
#     # "a": "あ",
#     # "hanyu": "방",
#     # "asdf": "д",
#     # "fds": "ห้",
#     # "smile":"☺",
#
# }
# word_list = [
#     "岁月静好",
#     "热爱",
#     "一帆风顺",
#     "Garry",
#     "今방天气不错啊"
# ]

file_name_list = [
    # "./tmp_example_img/mix_EN_character.txt",
    # "./tmp_example_img/mix_EN_word.txt",
    # "./tmp_example_img/mix_CN_character.txt",
    "./tmp_example_img/mix_CN_word.txt",
]
word_list = []
for it_file in file_name_list:
    with open(it_file, 'r') as f:
        for line in f.readlines():
            word_list.append(line.strip())

bg_color = (250, 248, 235)
fg_color = (85, 82, 79)

for i, text in enumerate(word_list):
    # check CN or EN
    text_to_draw = text.strip().split(' ')[0]
    if len(text_to_draw) == 1:
        # character
        if is_chinese(text_to_draw):
            default_font_path_list = CN_default_font_path
            font_path = CN_font_file_dict['QingNiaoxingkai']['file_path']
        else:
            default_font_path_list = EN_default_font_path
            font_path = EN_font_file_dict['HFPuff']['file_path']
        base_img = gen_character(text_to_draw, font_path, default_font_path_list, fg_color, bg_color)
    else:
        if text_to_draw.encode('utf-8').isalpha():
            default_font_path_list = EN_default_font_path
            font_path = EN_font_file_dict['HFPuff']['file_path']
            base_img = gen_word_EN(text_to_draw, font_path, default_font_path_list, fg_color, bg_color)
        else:
            text_to_draw = text_to_draw[:4]
            b_CN = True

            for it in text_to_draw:
                if not is_chinese(it):
                    b_CN = False
                    break
            if b_CN:
                default_font_path_list = CN_default_font_path
                font_path = CN_font_file_dict['QingNiaoxingkai']['file_path']
                base_img = gen_word_CN_together(text_to_draw, font_path, default_font_path_list, fg_color, bg_color)
            else:
                default_font_path_list = EN_default_font_path
                font_path = EN_font_file_dict['HFPuff']['file_path']
                base_img = gen_word_EN(text_to_draw, font_path, default_font_path_list, fg_color, bg_color)
    print(text_to_draw)
    save_name = os.path.join(save_path, text_to_draw + ".jpg")
    base_img.save(save_name)
