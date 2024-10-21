import sys
import os
import PIL.Image as Image

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "../../"))

from plugin_modules.ip_adapter.gen_baseword import compute_complex_word
from plugin_modules.ip_adapter.gen_basewordimg_v2 import b_complex_CN

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
    "wang": "王",
    "li": "李",
    "zhang": "张",
    "liu": "刘",
    "chen": "陈",
    "yang": "杨",
    "zhao": "赵",
    "huang": "黄",
    "zhou": "周",
    "wu": "吴",
    "xu": "徐",
    "sun": "孙",
    "hu": "胡",
    "zhu": "朱",
    "gao": "高",
    "lin": "林",
    "he": "何",
    "guo": "郭",
    "ma": "马",
    "luo": "罗",
    "liang": "梁",
    "song": "宋",
    "zheng": "郑",
    "xie": "谢",
    "han": "韩",
    "tang": "唐",
    "feng": "冯",
    "yu": "于",
    "dong": "董",
    "xiao": "萧",
    "cheng": "程",
    "cao": "曹",
    "yuan": "袁",
    "deng": "邓",
    "xu3": "许",
    "fu": "傅",
    "shen": "沈",
    "zeng": "曾",
    "peng": "彭",
    "lv": "吕",
    "su": "苏",
    "lu": "卢",
    "jiang": "蒋",
    "cai": "蔡",
    "jia": "贾",
    "wei": "魏",
    "dai": "戴",
}
word_list = [
    '王', '李', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴', '徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马',
    '罗', '梁', '宋', '郑', '谢', '韩', '唐', '冯', '于', '董', '萧', '程', '曹', '袁', '邓', '许', '傅', '沈', '曾',
    '彭', '吕', '苏', '卢', '蒋', '蔡', '贾', '丁', '魏', '薛', '叶', '阎',
    # 51,
    '余', '潘', '杜', '戴', '夏', '钟', '汪', '田', '任', '姜', '范', '方', '石', '姚', '谭', '廖', '邹', '熊', '金',
    '陆', '郝', '孔', '白', '崔', '康', '毛', '邱', '秦', '江', '史', '顾', '侯', '邵', '孟', '龙', '万', '段', '漕',
    '钱', '汤', '尹', '黎', '易', '常', '武', '乔', '贺', '赖', '龚', '文',
    # others
    '一', '丁', '七', '万', '丈', '三', '上', '下', '不', '与', '丐', '丑', '专', '且', '世', '丘', '丙',
    '业', '丛', '东', '丝', '丢', '两', '严', '丧', '个', '中', '丰', '串', '临', '丸', '丹', '为', '主',
    '丽', '举', '乃', '久', '么', '义', '之', '乌', '乍', '乎', '乏', '乐', '乒', '乓', '乔'
]

# baoli_result_dict = {}
# for it_word in word_list:
#     white_idx, black_idx = compute_complex_word(it_word, font_file_dict['baoli']['file_path'], font_size=812)
#     baoli_result_dict[it_word] = [white_idx, black_idx]
#
# xingkai_result_dict = {}
# for it_word in word_list:
#     white_idx, black_idx = compute_complex_word(it_word, font_file_dict['QingNiaoxingkai']['file_path'], font_size=812)
#     xingkai_result_dict[it_word] = [white_idx, black_idx]

baoli_result_dict = {}
for i, it_word in enumerate(word_list):
    ratio = b_complex_CN(it_word, font_file_dict['baoli']['file_path'],
                         default_font_path_list=[font_file_dict['baoli']['file_path']], font_size=812, text_ratio=0.85)
    baoli_result_dict[it_word] = ratio
    print("baoli", i, it_word)

xingkai_result_dict = {}
for i, it_word in enumerate(word_list):
    ratio = b_complex_CN(it_word, font_file_dict['QingNiaoxingkai']['file_path'],
                         default_font_path_list=[font_file_dict['QingNiaoxingkai']['file_path']], font_size=812,
                         text_ratio=0.85)
    xingkai_result_dict[it_word] = ratio
    print("QingNiaoxingkai", i, it_word)

baoli_cnt = 0
for k, v in baoli_result_dict.items():
    if v > 0.21:
        print("baoli: ", k, v)
        baoli_cnt += 1

xingkai_cnt = 0
for k, v in xingkai_result_dict.items():
    if v > 0.22:
        print("xingkai: ", k, v)
        xingkai_cnt += 1

print("complexity")
print("baoli:", baoli_cnt)
print("xingkai:", xingkai_cnt)
a = 0
