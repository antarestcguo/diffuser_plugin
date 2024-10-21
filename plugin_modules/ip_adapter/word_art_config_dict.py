# style_folder :{"prompt":,, "negative_prompt":,,,"font_style":,,,"font_size":,"bg_color",,,"fg_color":,,"strength":,,,}

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
    "liang": "梁",
    "xie": "谢",
    "dong": "董",
    "wei": "魏",
    "dai": "戴",
}

complexword_dict = {
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

negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, low quality, long neck, frame, text, worst quality,watermark, deformed, ugly,  blur, out of focus,extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, "
font_size = 812
enhance_type = "adapter"  # adapter or img2img
style_config_dict = {
    "butterflygirl_black":  # 0.75 0.8都可，refine提升脸部
        {
            "prompt": "a cartoon style gile with Long flowing black hair, blue butterfly on hair, full body photo, full-length portrait, watercolor,ink and wash, light clean background",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (250, 248, 235),
            "fg_color": (85, 82, 79),
            # char
            "strength": [0.8, 0.75],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "a cartoon style gile with Long flowing black hair, blue butterfly on hair, full body photo, full-length portrait, watercolor,ink and wash, light clean background, High quality, detail face,",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.85, 0.8],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.85, 0.8],
                    "word_EN_alpha_ratio": [0.85, 0.8],
                }
        },
    "male_black":  # 0.75和0.8都可，refine提升脸部
        {
            "prompt": "a cartoon style Martial arts Style man, male, in the bamboo forest,full body photo, full-length portrait, light clean background ",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (247, 248, 242),
            "fg_color": (101, 98, 96),
            "strength": [0.8, 0.75, ],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "a cartoon style Martial arts Style man, male, in the bamboo forest,full body photo, full-length portrait, light clean background ",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.85, 0.8],
                    "word_EN_alpha_ratio": [0.85, 0.8],
                }
        },
    "tree_black":  # 0.8更美，需要refine结构
        {
            "prompt": "log cabin with trees, lake, mountain, wooden boat, ink and wash, watercolor",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (252, 246, 234),
            "fg_color": (107, 116, 119),
            "strength": [0.8, 0.75],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "log cabin with trees, lake, mountain, wooden boat, ink and wash, watercolor",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    "winter_black":  # 0.75可以，但是0.8更美，0.8没结构，可以refine以下
        {
            "prompt": "log cabin with trees in snow, lake, mountain, plum blossom, wooden boat, ink and wash, watercolor",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['baoli'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (233, 237, 235),
            "fg_color": (86, 76, 81),
            "strength": [0.8, 0.75],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "log cabin with trees in snow, lake, mountain, plum blossom, wooden boat, ink and wash, watercolor",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },

    "dunhuanggirl_yellow":  # 0.75和0.8都可，但肢体容易乱，需要refine
        {
            "prompt": "Dunhuang style dancing female, Chinese tradition peri girl in yellow golden color,silk ribbon, dark background, detail face,",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (31, 66, 64),
            "fg_color": (243, 206, 159),
            "strength": [0.8, 0.75],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "Dunhuang style dancing female, Chinese tradition peri girl in yellow golden color,silk ribbon, High quality, detail face,",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },

    # "qianxun_black": # 不好看
    #     {
    #         "prompt": "cloud, one moon, Chinese ancient architecture, lake,lantern, boat on the lake",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": list(font_file_dict.keys()),
    #         "bg_color": (108, 138, 160),
    #         "fg_color": (38, 40, 59),
    #         "strength": [0.7, 0.75],
    #     },
    "shuimocartoon_green":  # 试一下0.8， 0.75不出逻辑
        {
            "prompt": "mountain with trees on the river, watercolor,ink and wash, aestheticism",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (229, 248, 233),
            "fg_color": (0, 120, 118),
            "strength": [0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "mountain with trees on the river, watercolor,ink and wash, aestheticism",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.8, 0.85],
                    "word_EN_alpha_ratio": [0.8, 0.85],
                }
        },
    "cartoongirl3_green":  # 试一下0.8
        {
            "prompt": "a cartoon style girl in green cloths, with lotus leaf and flowers,",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (199, 221, 209),
            "fg_color": (25, 85, 45),
            "strength": [0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "a cartoon style girl in green cloths, with lotus leaf and flowers,",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    # "gufenggirl_black":
    #     {
    #         "prompt": "Chinese ancient style woman in front of log cabin, red oil-paper umbrella, in winter with snow. ",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": list(font_file_dict.keys()),
    #         "bg_color": (181, 197, 201),
    #         "fg_color": (67, 64, 65),
    #     },
    "phoenix_white":  # 0.75更好看
        {
            "prompt": "white phoenix with rich plumage and beautiful wings",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (47, 74, 95),
            "fg_color": (215, 224, 226),
            "strength": [0.75],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "white phoenix with rich plumage and beautiful wings",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    # "wood_black": # 不好看，delete
    #     {
    #         "prompt": "building, scenery in misty, mountain log cabin, with trees and dim light",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": list(font_file_dict.keys()),
    #         "bg_color": (169, 182, 199),
    #         "fg_color": (54, 61, 77),
    #         "strength": [0.75],
    #     },
    "cartoongirl2_while":  # 0.7 0.75都留下，baoli和Qingniao似乎不同的strength
        {
            "prompt": "a cartoon style girl in white, white butterfly",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (149, 190, 196),
            "fg_color": (243, 251, 242),
            "strength": [0.7, 0.75],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "a cartoon style girl in white, white butterfly",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    "gufenggirl_white":  # 0.8会好一点，还是出现肢体问题，需要refine质量
        {
            "prompt": "Ancient Chinese drama style lady in Peking Opera headwear, white and blue cloth, rich detail in face",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (19, 48, 78),
            "fg_color": (232, 232, 232),
            "strength": [0.8, 0.75],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "Ancient Chinese drama style lady in Peking Opera headwear, white and blue cloth, rich detail in face",
                    "negative_prompt": negative_prompt,
                    "strength": [0.75],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    # "treewoodcartoon_green":
    #     {
    #         "prompt": "cartoon style landscape painting in green, log cabin with trees mountain, Waterfall and flowing water",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": list(font_file_dict.keys()),
    #         "bg_color": (230, 230, 217),
    #         "fg_color": (63, 82, 108),
    #     },
    # "fengjingwood_black":  # 先留着，不是特别好看,准备删了，还是不好看
    #     {
    #         "prompt": "log cabin, scenery in misty, mountain, with trees,",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": list(font_file_dict.keys()),
    #         "bg_color": (214, 219, 222),
    #         "fg_color": (70, 88, 100),
    #         "strength": [0.75],
    #         "enhance":
    #             {
    #                 "prompt": "log cabin, scenery in misty, mountain, with trees,",
    #                 "negative_prompt": negative_prompt,
    #                 "strength": [0.7,0.75],
    #             }
    #     },
    "shuimocartoonfox_black":  # 0.7不好看
        {
            "prompt": "watercolor,ink and wash, cartoon style, trees and fox",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (246, 244, 231),
            "fg_color": (141, 128, 122),
            "strength": [0.75],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "watercolor,ink and wash, cartoon style, trees and fox",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],

                }
        },
    "shuimocartoongirl2_green":  # 再试一次0.8
        {
            "prompt": "a girl in watercolor,ink and wash with lotus leaf, flowers",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (247, 243, 231),
            "fg_color": (101, 129, 118),
            "strength": [0.75],
            "EN_strength": [0.8],
            "enhance":
                {
                    "prompt": "a girl in watercolor,ink and wash with lotus leaf, flowers",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.7, ],
                    "word_EN_strength": [0.7, ],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    # "god_white":
    #     {
    #         "prompt": "Chinese the God of Wealth in gold color, white mustache and cloud, rich details in face, gold coins, flowers, Tower log house",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": list(font_file_dict.keys()),
    #         "bg_color": (176, 116, 66),
    #         "fg_color": (243, 244, 235),
    #     },
    "shuimowood2_black":  # 0.8 需要尝试refine，0.8结构没了
        {
            "prompt": "log cabin with trees, lake, mountain, lotus leaf, flowers,Stone staircase, ink and wash, watercolor",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (250, 241, 225),
            "fg_color": (98, 104, 114),
            "strength": [0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "log cabin with trees, lake, mountain, lotus leaf, flowers,Stone staircase, ink and wash, watercolor",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    "shuimowood_black":  # 0.8 需要尝试refine，0.8结构没了
        {
            "prompt": "log cabin with trees, lake, mountain, wooden boat,Stone staircase, ink and wash, watercolor",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (252, 248, 236),
            "fg_color": (101, 114, 118),
            "strength": [0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "log cabin with trees, lake, mountain, wooden boat,Stone staircase, ink and wash, watercolor",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    # "phoenix_red":  # 0.75不行丢失结构, 暂时放弃，不更新不维护
    #     {
    #         "prompt": "red phoenix with rich plumage and beautiful wing",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": ['QingNiaoxingkai'],
    #         "EN_font_style": ['HFPuff'],
    #         "bg_color": (229, 200, 151),
    #         "fg_color": (139, 76, 41),
    #         "strength": [0.7],
    #         "EN_strength": [ 0.85],
    #         "enhance":
    #             {
    #                 "prompt": "red phoenix with rich plumage and beautiful wing",
    #                 "negative_prompt": negative_prompt,
    #                 "strength": [0.7, 0.75],
    #                 "alpha_ratio": [0.8, 0.9],
    #             }
    #     },
    # "prince_black": # 肢体问题大,delete
    #     {
    #         "prompt": "cartoon style boy with fox on the grass,watercolor,ink and wash style",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": list(font_file_dict.keys()),
    #         "bg_color": (224, 236, 228),
    #         "fg_color": (45, 52, 64),
    #         "strength": [0.8],
    #     },
    "tiankongzhicheng_black":  # 0.8更好，refine以下结构
        {
            "prompt": "Miyazaki style, castle in sky,log cabin with trees, lake, Stone staircase, Surrounded by clouds. great moon, flower",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (196, 209, 207),
            "fg_color": (59, 67, 76),
            "strength": [0.8, 0.75],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "Miyazaki style, castle in sky,log cabin with trees, lake, Stone staircase, Surrounded by clouds. great moon, flower",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.8, 0.85],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    "shuimocartoongirl_black":  # 0.75 0.8都可以，复杂字需要0.75，不然容易没有结构
        {
            "prompt": "cartoon style girl with blue red black cloths,flower, lantern,watercolor,ink and wash,",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (228, 226, 213),
            "fg_color": (57, 84, 130),
            "strength": [0.75, 0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "cartoon style girl with blue red black cloths,flower, lantern,watercolor,ink and wash,",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    # "shuimohuizhou_black": # 不好看，删除
    #     {
    #         "prompt": "Hui style architecture, house building,river, stone,ink and wash,Chinese paintings",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": list(font_file_dict.keys()),
    #         "bg_color": (232, 227, 217),
    #         "fg_color": (30, 29, 27),
    #         "strength": [0.75],
    #     },
    "shuimoflower_black":  # 再试一次0.8
        {
            "prompt": "watercolor,ink and wash, flower and butterfly, Chinese paintings",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (243, 234, 219),
            "fg_color": (48, 58, 52),
            "strength": [0.75, 0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "watercolor,ink and wash, flower and butterfly, Chinese paintings",
                    "negative_prompt": negative_prompt,
                    "strength": [0.8, 0.75],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.8, 0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    "cartoongirl_black":  # 0.75 和 0.8 都可，0.8好看，会丢失结构
        {
            "prompt": "a cartoon girl in black cloth, black hair, wings in background, light background, rich detail in face,",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (230, 221, 214),
            "fg_color": (64, 74, 75),
            "strength": [0.75, 0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "a cartoon girl in black cloth, black hair, wings in background, light background, rich detail in face,",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.85, 0.9],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    # "woodboat_black":  # 0.65 不好看，再网上又没有结构了，暂时删
    #     {
    #         "prompt": "log cabin with trees, river, mountain, wooden boat, ink and wash, watercolor, cartoon style",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": list(font_file_dict.keys()),
    #         "bg_color": (214, 211, 214),
    #         "fg_color": (99, 111, 138),
    #         "strength": [0.65],
    #     },

    # add from stylist

    # add from stylist
    "shuimodesigner_black":  # 0.75 和 0.8 逻辑没了，向0.7 0.6 去试试, 0.65定下来，0.6完全没有画面
        {
            "prompt": "chinese ink painting, mountain, trees, Stone step,log cabin",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (200, 180, 162),
            "fg_color": (112, 100, 100),
            "strength": [0.65],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "chinese ink painting, mountain, trees, Stone step,log cabin",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.65],
                    "alpha_ratio": [0.85, 0.9],
                    # word
                    "word_CN_strength": [0.7, ],
                    "word_EN_strength": [0.7, ],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    ################## idx:19 (0-18)
    # "male2_black": #和前面风格很重复，不要了
    #     {
    #         "prompt": "a cartoon style Martial arts Style man, male, in the bamboo forest, ,full body photo, full-length portrait, light clean background ",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": ['QingNiaoxingkai'],
    #         "EN_font_style": ['HFPuff'],
    #         "bg_color": (215, 215, 211),
    #         "fg_color": (79, 89, 95),
    #         "strength": [0.75, 0.8],
    #         "EN_strength": [ 0.85],
    #         "enhance":
    #             {
    #                 "prompt": "a cartoon style Martial arts Style man, male, in the bamboo forest, ,full body photo, full-length portrait, light clean background ",
    #                 "negative_prompt": negative_prompt,
    #                 "strength": [0.7, 0.75],
    #                 "alpha_ratio": [0.8, 0.85],
    #             }
    #     },
    "girl_blue":
        {
            "prompt": "Ancient Chinese drama style lady in blue and white skirt, clean background",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (112, 111, 128),
            "fg_color": (225, 228, 235),
            "strength": [0.75, 0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "Ancient Chinese drama style lady in blue and white skirt, clean background",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.5],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    # "girlsky_blue": #人脸面部结构不好，放弃
    #     {
    #         "prompt": "a lady in blue and white skirt, background is blue sky and white cloud",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": ['QingNiaoxingkai'],
    #         "EN_font_style": ['HFPuff'],
    #         "bg_color": (159, 190, 208),
    #         "fg_color": (33, 85, 133),
    #         "strength": [0.75, 0.8],
    #         "EN_strength": [ 0.85],
    #         "enhance":
    #             {
    #                 "prompt": "a lady in blue and white skirt, background is blue sky and white cloud",
    #                 "negative_prompt": negative_prompt,
    #                 "strength": [0.7, 0.75],
    #                 "alpha_ratio": [0.8, 0.85],
    #             }
    #     },
    # "girlcrane_white": # 生成图不好看，放弃
    #     {
    #         "prompt": "a lady in blue and white skirt, background is blue sky and white cloud",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": ['QingNiaoxingkai'],
    #         "EN_font_style": ['HFPuff'],
    #         "bg_color": (16, 38, 26),
    #         "fg_color": (210, 212, 214),
    #         "strength": [0.75, 0.8],
    #         "EN_strength": [ 0.85],
    #         "enhance":
    #             {
    #                 "prompt": "a lady in black and white skirt, with white crane bird, dark background",
    #                 "negative_prompt": negative_prompt,
    #                 "strength": [0.7, 0.75],
    #                 "alpha_ratio": [0.8, 0.85],
    #             }
    #     },
    # "dog_light": # 猫和狗，逻辑都差，出图率低，暂时放弃
    #     {
    #         "prompt": "a dog with light, dark background",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": ['QingNiaoxingkai'],
    #         "EN_font_style": ['HFPuff'],
    #         "bg_color": (17, 19, 20),
    #         "fg_color": (232, 215, 199),
    #         "strength": [0.75, 0.8],
    #         "EN_strength": [ 0.85],
    #         "enhance":
    #             {
    #                 "prompt": "a dog with light,dark background",
    #                 "negative_prompt": negative_prompt,
    #                 "strength": [0.7, 0.75],
    #                 "alpha_ratio": [0.8, 0.85],
    #             }
    #     },
    # "cat_light": # 猫和狗，逻辑都差，出图率低，暂时放弃
    #     {
    #         "prompt": "a cat with light, dark background",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": ['QingNiaoxingkai'],
    #         "EN_font_style": ['HFPuff'],
    #         "bg_color": (17, 19, 20),
    #         "fg_color": (232, 215, 199),
    #         "strength": [0.75, 0.8],
    #         "EN_strength": [ 0.85],
    #         "enhance":
    #             {
    #                 "prompt": "a cat with light,dark background",
    #                 "negative_prompt": negative_prompt,
    #                 "strength": [0.7, 0.75],
    #                 "alpha_ratio": [0.8, 0.85],
    #             }
    #     },
    "night_white":
        {
            "prompt": "night moon, mountain, stone bridge, wooden house,trees, white cloud and white waterfall",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (76, 72, 72),
            "fg_color": (234, 237, 230),
            "strength": [0.75, 0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "night moon, mountain, stone bridge, wooden house,trees, white cloud and white waterfall",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    "flower_white":  # 降低 strength试试，第二次统一try的时候试
        {
            "prompt": "white flowers and green leaves,  photographic, intricate detail, Ultra Detailed hyperrealistic real photo, 8k, high quality,",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (53, 85, 48),
            "fg_color": (183, 191, 166),
            "strength": [0.75, 0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "white flowers and green leaves,  photographic, intricate detail, Ultra Detailed hyperrealistic real photo, 8k, high quality,",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.8, 0.85],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    "fengjingcartoon_black":  # 降低 strength试试，第二次统一try的时候试
        {
            "prompt": "cartoon, wooden house, trees, rivers",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (179, 192, 184),
            "fg_color": (83, 83, 96),
            "strength": [0.7],
            "EN_strength": [0.8],
            "enhance":
                {
                    "prompt": "cartoon, wooden house, trees, rivers",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.7],
                    "word_EN_strength": [0.7],
                    "word_CN_alpha_ratio": [0.8, 0.85],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    # "cartoongirl4_black": #不好看，和前面风格也比较重复
    #     {
    #         "prompt": "cartoon style girl,lotus and leaf, watercolor,ink and wash,clean background",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": ['QingNiaoxingkai'],
    #         "EN_font_style": ['HFPuff'],
    #         "bg_color": (240, 236, 227),
    #         "fg_color": (12, 46, 39),
    #         "strength": [0.75, 0.8],
    #         "EN_strength": [ 0.85],
    #         "enhance":
    #             {
    #                 "prompt": "cartoon style girl,lotus and leaf, watercolor,ink and wash,clean background",
    #                 "negative_prompt": negative_prompt,
    #                 "strength": [0.7, 0.75],
    #                 "alpha_ratio": [0.8, 0.85],
    #             }
    #     },
    "male3_black":
        {
            "prompt": "cartoon style male, white and black cloth,full body photo, full-length portrait, light clean background ",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (197, 211, 226),
            "fg_color": (45, 69, 96),
            "strength": [0.75, 0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "cartoon style male, white and black cloth,full body photo, full-length portrait, light clean background ",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.75],
                    "word_EN_strength": [0.75],
                    "word_CN_alpha_ratio": [0.5],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    "fengjing_white":  # 降低 strength试试，第二次统一try的时候试
        {
            "prompt": "lake, wooden house, mountain, trees,white cloud and white inverted reflection in water",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (142, 160, 170),
            "fg_color": (243, 244, 238),
            "strength": [0.75],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "lake, wooden house, mountain, trees,white cloud and white inverted reflection in water",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.7],
                    "word_EN_strength": [0.7],
                    "word_CN_alpha_ratio": [0.8, 0.85],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    "magiccartoongirl_black":
        {
            "prompt": "magic cartoon girl,black and white cloth, long hair",
            "negative_prompt": negative_prompt,
            "font_size": font_size,
            "font_style": ['QingNiaoxingkai'],
            "EN_font_style": ['HFPuff'],
            "bg_color": (126, 159, 170),
            "fg_color": (23, 32, 39),
            "strength": [0.75, 0.8],
            "EN_strength": [0.85],
            "enhance":
                {
                    "prompt": "magic cartoon girl,black and white cloth, long hair",
                    "negative_prompt": negative_prompt,
                    "strength": [0.7, 0.75],
                    "alpha_ratio": [0.8, 0.85],
                    # word
                    "word_CN_strength": [0.7],
                    "word_EN_strength": [0.7],
                    "word_CN_alpha_ratio": [0.0],
                    "word_EN_alpha_ratio": [0.0],
                }
        },
    #     "shuimogirl_green": # 不好看，画面太单调，放弃
    # {
    #             "prompt": "a cartoon girl, trees, rich detail in face,watercolor,ink and wash, aestheticism",
    #             "negative_prompt": negative_prompt,
    #             "font_size": font_size,
    #             "font_style": ['QingNiaoxingkai'],
    #             "EN_font_style": ['HFPuff'],
    #             "bg_color": (242,254,254),
    #             "fg_color": (20,138,135),
    #             "strength": [0.75, 0.8],
    #             "EN_strength": [ 0.85],
    #             "enhance":
    #                 {
    #                     "prompt": "a cartoon girl, trees, rich detail in face,watercolor,ink and wash, aestheticism",
    #                     "negative_prompt": negative_prompt,
    #                     "strength": [0.7, 0.75],
    #                     "alpha_ratio": [0.8, 0.85],
    #                 }
    #         },

    # "girl_real":
    #     {
    #         "prompt": "girl,black and white cloth, black long hair, masterpiece, photographic, intricate detail, Ultra Detailed hyperrealistic real photo",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": ['QingNiaoxingkai'],
    #         "EN_font_style": ['HFPuff'],
    #         "bg_color": (165, 187, 173),
    #         "fg_color": (53, 54, 72),
    #         "strength": [0.75, 0.8],
    #         "EN_strength": [0.85],
    #         "enhance":
    #             {
    #                 "prompt": "girl,black and white cloth, black long hair, masterpiece, photographic, intricate detail, Ultra Detailed hyperrealistic real photo",
    #                 "negative_prompt": negative_prompt,
    #                 "strength": [0.7, 0.75],
    #                 "alpha_ratio": [0.8, 0.85],
    #                 # word
    #                 "word_CN_strength": [0.7],
    #                 "word_EN_strength": [0.7],
    #                 "word_CN_alpha_ratio": [0.0],
    #                 "word_EN_alpha_ratio": [0.0],
    #             }
    #     },
    #
    # "cat_real":
    #     {
    #         "prompt": "cute cat, background is mountain, cloud, sky, sun, masterpiece, photographic, intricate detail, Ultra Detailed hyperrealistic real photo",
    #         "negative_prompt": negative_prompt,
    #         "font_size": font_size,
    #         "font_style": ['QingNiaoxingkai'],
    #         "EN_font_style": ['HFPuff'],
    #         "bg_color": (165, 187, 173),
    #         "fg_color": (53, 54, 72),
    #         "strength": [0.75, 0.8],
    #         "EN_strength": [0.85],
    #         "enhance":
    #             {
    #                 "prompt": "cute cat, background is mountain, cloud, sky, sun, masterpiece, photographic, intricate detail, Ultra Detailed hyperrealistic real photo",
    #                 "negative_prompt": negative_prompt,
    #                 "strength": [0.7, 0.75],
    #                 "alpha_ratio": [0.8, 0.85],
    #                 # word
    #                 "word_CN_strength": [0.7],
    #                 "word_EN_strength": [0.7],
    #                 "word_CN_alpha_ratio": [0.0],
    #                 "word_EN_alpha_ratio": [0.0],
    #             }
    #     },
}
