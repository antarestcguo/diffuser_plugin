from PIL import Image, ImageDraw, ImageFont, ImageChops
import cv2
import numpy as np
from fontTools.ttLib import TTFont

complex_EN_param = 5
complex_CN_param = 0.22
word_space_ratio_CN = 0.2 # 0.1
base_pixel_number = 8


def is_chinese(char):
    # Check if a character is Chinese
    return '\u4e00' <= char <= '\u9fff'


def gen_img_grey(text_to_draw, font, text_ratio):  # return cv2.img pad_grey_binary,crop_grey_binary
    # no matther text_to_draw is character or word
    text_width, text_height = font.getsize(text_to_draw)
    # cv2 compute bbox
    image_text_grey = Image.new("RGB", (text_width, text_height), (255, 255, 255))  # RGB, not RGBA
    draw_grey = ImageDraw.Draw(image_text_grey)
    draw_grey.text((0, 0), text_to_draw, font=font, fill=(0, 0, 0))
    # convert 2 opencv , compute the min bbox
    cv_img = cv2.cvtColor(np.asarray(image_text_grey), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(cv_img, 127, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(255 - binary)
    # compute final image w and h
    if len(text_to_draw) == 1:
        text_max_side = max(w, h)
        img_w = img_h = int(float(text_max_side) / text_ratio)
    else:
        img_w = int(float(w) / text_ratio)
        img_h = int(float(h) / text_ratio)

    pad_image = np.ones((img_h, img_w), dtype=np.uint8) * 255
    start_x = int((img_w - w) / 2)
    start_y = int((img_h - h) / 2)
    pad_image[start_y:start_y + h, start_x:start_x + w] = binary[y:y + h, x:x + w]

    return pad_image, binary[y:y + h, x:x + w]


def fill_color(binary_cvimg, bg_color, fg_color, color_img=None):  # if RGBA, process bg/fg color outside
    # fill fg/bg color
    h, w = binary_cvimg.shape
    if color_img is None:
        color_img = np.zeros([h, w, 3], dtype=np.uint8)
    if bg_color is not None:
        color_img[(binary_cvimg == 255), :] = bg_color
    color_img[(binary_cvimg == 0), :] = fg_color

    return color_img


def polybox2points(box):
    points = []
    for it in box:
        points.append(tuple(it[0]))

    return points


def b_text_can_be_draw(font_path, text_to_draw):
    b_draw = False
    font = TTFont(font_path)

    glyph_name = None
    for table in font['cmap'].tables:
        glyph_name = table.cmap.get(ord(text_to_draw))
        if glyph_name is not None:
            break

    if glyph_name is not None:
        glyf = font['glyf']
        if glyf.has_key(glyph_name):
            a = glyf[glyph_name].getCoordinates(0)[0]
            if len(a) > 0:
                b_draw = True
    return b_draw


def obtain_draw_font(text, font_path, default_font_path_list):
    b_draw = True
    select_font_path = font_path
    for it_text in text:
        b_draw = b_text_can_be_draw(font_path, it_text)
        if not b_draw:
            for it_default_font_path in default_font_path_list:
                b_draw = b_text_can_be_draw(it_default_font_path, it_text)
                if b_draw:
                    select_font_path = it_default_font_path
                    break
        if not b_draw:
            return None
    return select_font_path


def b_complex_EN(text):
    if len(text) > complex_EN_param:
        return True


def b_complex_CN(text, font_path, default_font_path_list, font_size, text_ratio):
    select_font_path = obtain_draw_font(text, font_path, default_font_path_list)
    if select_font_path is None:
        return None

    font = ImageFont.truetype(select_font_path, font_size)
    _, crop_grey = gen_img_grey(text, font, text_ratio)
    text_width, text_height = font.getsize(text)

    idx = np.sum(crop_grey == 0)
    ratio = float(idx) / text_width / text_height

    if ratio > complex_CN_param:
        return True
    else:
        return False


def gen_character(text, font_path, default_font_path_list, fg_color, bg_color, font_size=812, text_ratio=0.85,
                  max_side=1024, min_side=1024):
    # determin if can be draw
    select_font_path = obtain_draw_font(text, font_path, default_font_path_list)
    if select_font_path is None:
        return None

    # draw binary_grey img
    font = ImageFont.truetype(select_font_path, font_size)
    pad_grey, crop_grey = gen_img_grey(text, font, text_ratio)
    # fill color
    pad_color = fill_color(pad_grey, bg_color, fg_color)

    # convert to PIL Image and resize
    img = Image.fromarray(pad_color)
    final_img = img.resize((max_side, max_side), Image.BILINEAR)
    return final_img


def gen_word_EN(text, font_path, default_font_path_list, fg_color, bg_color, font_size=812, text_ratio=0.85,
                max_side=2048, min_side=512):
    # determin if can be draw
    select_font_path = obtain_draw_font(text, font_path, default_font_path_list)
    if select_font_path is None:
        return None

    # draw binary_grey img
    font = ImageFont.truetype(select_font_path, font_size)
    pad_grey, crop_grey = gen_img_grey(text, font, text_ratio)

    # fill color
    pad_color = fill_color(pad_grey, bg_color, fg_color)

    # convert to PIL Image and resize
    img = Image.fromarray(pad_color)
    img_w, img_h = img.size

    ratio = min_side / min(img_h, img_w)
    img_w, img_h = round(ratio * img_w), round(ratio * img_h)
    ratio = max_side / max(img_h, img_w)
    final_img = img.resize((round(ratio * img_w), round(ratio * img_h)), Image.BILINEAR)

    return final_img


def gen_word_CN_together(text, font_path, default_font_path_list, fg_color, bg_color, font_size=812, text_ratio=0.85,
                         max_side=2048, min_side=512):  # return one image together
    # determin if can be draw
    select_font_path = obtain_draw_font(text, font_path, default_font_path_list)
    if select_font_path is None:
        return None

    # compute binary_grey img list
    crop_grey_list = []
    character_width_list = []
    character_height_list = []
    font = ImageFont.truetype(select_font_path, font_size)
    for it in text:
        pad_grey, crop_grey = gen_img_grey(it, font, text_ratio)
        crop_grey_list.append(crop_grey)
        h, w = crop_grey.shape
        character_width_list.append(w)
        character_height_list.append(h)

    # compute merge img w and h
    merge_h = np.max(character_height_list)
    merge_w = character_width_list[0]
    for i in range(1, len(character_width_list)):
        merge_w = merge_w + character_width_list[i - 1] * word_space_ratio_CN + character_width_list[i]
    pad_merge_h = int(merge_h / text_ratio)
    pad_merge_w = int(merge_w / text_ratio)

    # merge all the crop grey image
    merge_img = np.ones([pad_merge_h, pad_merge_w], dtype=np.uint8) * 255
    start_x = int((pad_merge_w - merge_w) / 2)
    for i, it in enumerate(crop_grey_list):
        start_y = int((pad_merge_h - character_height_list[i]) / 2)
        merge_img[start_y:start_y + character_height_list[i], start_x:start_x + character_width_list[i]] = it
        start_x = int(start_x + character_width_list[i] * (1.0 + word_space_ratio_CN))

    # fill color
    pad_color = fill_color(merge_img, bg_color, fg_color)

    # convert to PIL Image and resize
    img = Image.fromarray(pad_color)
    img_w, img_h = img.size

    ratio = min_side / min(img_h, img_w)
    img_w, img_h = round(ratio * img_w), round(ratio * img_h)
    ratio = max_side / max(img_h, img_w)
    final_img = img.resize((round(ratio * img_w), round(ratio * img_h)), Image.BILINEAR)
    return final_img


def gen_word_CN_separate(text, font_path, default_font_path_list, fg_color, bg_color, font_size=812, text_ratio=0.85,
                         max_side=2048, min_side=512):
    select_font_path = obtain_draw_font(text, font_path, default_font_path_list)
    if select_font_path is None:
        return None

    font = ImageFont.truetype(select_font_path, font_size)
    tmp_crop_list = []
    tmp_character_width_list = []
    tmp_character_height_list = []
    for it in text:
        _, crop_grey = gen_img_grey(it, font, text_ratio)
        tmp_crop_list.append(crop_grey)
        h, w = crop_grey.shape
        tmp_character_width_list.append(w)
        tmp_character_height_list.append(h)

    merge_w = tmp_character_width_list[0]
    for i in range(1, len(tmp_character_width_list)):
        merge_w = merge_w + tmp_character_width_list[i - 1] * word_space_ratio_CN + tmp_character_width_list[i]
    merge_h = np.max(tmp_character_height_list)

    # compute ratio
    if float(merge_w / merge_h) > 4:
        text_ratio_x = text_ratio
        text_ratio_y = text_ratio - 0.1
    else:
        text_ratio_x = text_ratio
        text_ratio_y = text_ratio

    pad_merge_h = int(merge_h / text_ratio_y)
    pad_merge_w = int(merge_w / text_ratio_x)

    # merge all the crop grey image
    merge_img = np.ones([pad_merge_h, pad_merge_w], dtype=np.uint8) * 255
    start_x = int((pad_merge_w - merge_w) / 2)
    offset_x = int((pad_merge_w - merge_w) / 2)
    for i, it in enumerate(tmp_crop_list):
        start_y = int((pad_merge_h - tmp_character_height_list[i]) / 2)
        merge_img[start_y:start_y + tmp_character_height_list[i], start_x:start_x + tmp_character_width_list[i]] = it
        start_x = int(start_x + tmp_character_width_list[i] * (1.0 + word_space_ratio_CN))

    # fill color
    pad_color = fill_color(merge_img, bg_color, fg_color)

    # convert to PIL Image and resize
    img = Image.fromarray(pad_color)
    img_w, img_h = img.size

    ratio = min_side / min(img_h, img_w)
    img_w, img_h = round(ratio * img_w), round(ratio * img_h)
    ratio = max_side / max(img_h, img_w)
    final_img = img.resize((round(ratio * img_w), round(ratio * img_h)), Image.BILINEAR)

    final_w, final_h = final_img.size

    resize_ratio = float(final_h) / img.size[1]

    # resize crop img
    crop_list = []
    tmp_resize_character_width_list = []
    # tmp_resize_character_height_list = []
    for i, it in enumerate(tmp_crop_list):
        crop_h, crop_w = tmp_character_height_list[i], tmp_character_width_list[i]
        resize_crop_h, resize_crop_w = int(crop_h * resize_ratio), int(crop_w * resize_ratio)
        tmp_resize_character_width_list.append(resize_crop_w)
        # tmp_resize_character_height_list.append(resize_crop_h)
        crop_list.append(cv2.resize(it, (resize_crop_w, resize_crop_h)))

    # slice final image
    slice_final_list = []
    end_x = int(offset_x * resize_ratio + tmp_resize_character_width_list[0] * (1 + word_space_ratio_CN / 2))
    slice_final_list.append(final_img.crop((0, 0, end_x, final_h)))
    start_x = end_x
    for i in range(1, len(crop_list) - 1):
        end_x = int(start_x + tmp_resize_character_width_list[i - 1] * word_space_ratio_CN / 2 +
                    tmp_resize_character_width_list[i] * (1 + word_space_ratio_CN / 2))
        slice_final_list.append(final_img.crop((start_x, 0, end_x, final_h)))
        start_x = end_x
    slice_final_list.append(final_img.crop((start_x, 0, final_w, final_h)))

    return slice_final_list, crop_list, text_ratio_x, text_ratio_y


def gen_word_CN_multidiffusion(text, font_path, default_font_path_list, fg_color, bg_color, font_size=812,
                               text_ratio=0.85,
                               max_side=4096, min_side=1024):
    select_font_path = obtain_draw_font(text, font_path, default_font_path_list)
    if select_font_path is None:
        return None

    font = ImageFont.truetype(select_font_path, font_size)
    tmp_crop_list = []
    tmp_character_width_list = []
    tmp_character_height_list = []
    for it in text:
        _, crop_grey = gen_img_grey(it, font, text_ratio)
        tmp_crop_list.append(crop_grey)
        h, w = crop_grey.shape
        tmp_character_width_list.append(w)
        tmp_character_height_list.append(h)

    merge_w = tmp_character_width_list[0]
    for i in range(1, len(tmp_character_width_list)):
        merge_w = merge_w + tmp_character_width_list[i - 1] * word_space_ratio_CN + tmp_character_width_list[i]
    merge_h = np.max(tmp_character_height_list)

    # compute ratio
    if float(merge_w / merge_h) > 4:
        text_ratio_x = text_ratio
        text_ratio_y = text_ratio - 0.1
    else:
        text_ratio_x = text_ratio
        text_ratio_y = text_ratio

    pad_merge_h = int(merge_h / text_ratio_y)
    pad_merge_w = int(merge_w / text_ratio_x)

    # merge all the crop grey image
    merge_img = np.ones([pad_merge_h, pad_merge_w], dtype=np.uint8) * 255
    start_x = int((pad_merge_w - merge_w) / 2)
    offset_x = int((pad_merge_w - merge_w) / 2)
    for i, it in enumerate(tmp_crop_list):
        start_y = int((pad_merge_h - tmp_character_height_list[i]) / 2)
        merge_img[start_y:start_y + tmp_character_height_list[i], start_x:start_x + tmp_character_width_list[i]] = it
        start_x = int(start_x + tmp_character_width_list[i] * (1.0 + word_space_ratio_CN))

    # fill color
    pad_color = fill_color(merge_img, bg_color, fg_color)

    # convert to PIL Image and resize
    img = Image.fromarray(pad_color)
    img_w, img_h = img.size

    ratio = min_side / min(img_h, img_w)
    img_w, img_h = round(ratio * img_w), round(ratio * img_h)
    ratio = max_side / max(img_h, img_w)
    if ratio > 1:
        final_w = img_w // base_pixel_number * base_pixel_number
        final_h = img_h // base_pixel_number * base_pixel_number
    else:
        final_w = round(ratio * img_w) // base_pixel_number * base_pixel_number
        final_h = round(ratio * img_h) // base_pixel_number * base_pixel_number
    final_img = img.resize((final_w, final_h), Image.BILINEAR)

    resize_ratio = float(final_h) / img.size[1]

    # resize crop img
    crop_list = []
    tmp_resize_character_width_list = []
    # tmp_resize_character_height_list = []
    for i, it in enumerate(tmp_crop_list):
        crop_h, crop_w = tmp_character_height_list[i], tmp_character_width_list[i]
        resize_crop_h, resize_crop_w = int(crop_h * resize_ratio // base_pixel_number * base_pixel_number), int(
            crop_w * resize_ratio // base_pixel_number * base_pixel_number)
        tmp_resize_character_width_list.append(resize_crop_w)
        # tmp_resize_character_height_list.append(resize_crop_h)
        crop_list.append(cv2.resize(it, (resize_crop_w, resize_crop_h)))

    # resize offset
    resize_offset_x = int(offset_x * resize_ratio // base_pixel_number * base_pixel_number)

    # slice views (h_start, h_end, w_start, w_end)
    slice_view_list = []
    end_x = int(resize_offset_x + tmp_resize_character_width_list[0] * (
            1 + word_space_ratio_CN * 2)) // base_pixel_number * base_pixel_number

    slice_view_list.append((0, final_h // base_pixel_number, 0, end_x // base_pixel_number))
    start_x = end_x - int(word_space_ratio_CN * 2 * tmp_resize_character_width_list[0])
    for i in range(1, len(crop_list) - 1):
        end_x = int(start_x + tmp_resize_character_width_list[i - 1] * word_space_ratio_CN * 2 +
                    tmp_resize_character_width_list[i] * (
                            1 + word_space_ratio_CN * 2)) // base_pixel_number * base_pixel_number
        slice_view_list.append(
            (0, final_h // base_pixel_number, start_x // base_pixel_number, end_x // base_pixel_number))
        start_x = end_x - int(word_space_ratio_CN * 2 * tmp_resize_character_width_list[i])
    slice_view_list.append(
        (0, final_h // base_pixel_number, start_x // base_pixel_number, final_w // base_pixel_number))

    # test
    # slice_view_list = []
    # panorama_height = int(final_h / base_pixel_number)
    # panorama_width = int(final_w / base_pixel_number)
    # window_size = 128
    # stride = 64
    #
    # num_blocks_width = (panorama_width - window_size) // stride + 1 if panorama_width > window_size else 1
    # num_blocks_width = int(num_blocks_width)
    #
    # for i in range(num_blocks_width):
    #     w_start = int(i * stride)
    #     w_end = w_start + window_size
    #     slice_view_list.append((0, panorama_height, w_start, w_end))

    return final_img, slice_view_list, crop_list, text_ratio_x, text_ratio_y, resize_offset_x


def gen_word_CN_separate_multidiffusion(text, font_path, default_font_path_list, fg_color, bg_color, font_size=812,
                                        text_ratio=0.85,
                                        max_side=4096, min_side=1024): # select this method into clean
    select_font_path = obtain_draw_font(text, font_path, default_font_path_list)
    if select_font_path is None:
        return None

    font = ImageFont.truetype(select_font_path, font_size)
    tmp_crop_list = []
    tmp_character_width_list = []
    tmp_character_height_list = []
    for it in text:
        _, crop_grey = gen_img_grey(it, font, text_ratio)
        tmp_crop_list.append(crop_grey)
        h, w = crop_grey.shape
        tmp_character_width_list.append(w)
        tmp_character_height_list.append(h)

    merge_w = tmp_character_width_list[0]
    for i in range(1, len(tmp_character_width_list)):
        merge_w = merge_w + tmp_character_width_list[i - 1] * word_space_ratio_CN + tmp_character_width_list[i]
    merge_h = np.max(tmp_character_height_list)

    # compute ratio
    if float(merge_w / merge_h) > 4:
        text_ratio_x = text_ratio
        text_ratio_y = text_ratio - 0.1
    else:
        text_ratio_x = text_ratio
        text_ratio_y = text_ratio

    pad_merge_h = int(merge_h / text_ratio_y)
    pad_merge_w = int(merge_w / text_ratio_x)

    # merge all the crop grey image
    merge_img = np.ones([pad_merge_h, pad_merge_w], dtype=np.uint8) * 255
    start_x = int((pad_merge_w - merge_w) / 2)
    offset_x = int((pad_merge_w - merge_w) / 2)
    for i, it in enumerate(tmp_crop_list):
        start_y = int((pad_merge_h - tmp_character_height_list[i]) / 2)
        merge_img[start_y:start_y + tmp_character_height_list[i], start_x:start_x + tmp_character_width_list[i]] = it
        start_x = int(start_x + tmp_character_width_list[i] * (1.0 + word_space_ratio_CN))

    # fill color
    pad_color = fill_color(merge_img, bg_color, fg_color)

    # convert to PIL Image and resize
    img = Image.fromarray(pad_color)
    img_w, img_h = img.size

    ratio = min_side / min(img_h, img_w)
    img_w, img_h = round(ratio * img_w), round(ratio * img_h)
    ratio = max_side / max(img_h, img_w)
    if ratio > 1:
        final_w = img_w // base_pixel_number * base_pixel_number
        final_h = img_h // base_pixel_number * base_pixel_number
    else:
        final_w = round(ratio * img_w) // base_pixel_number * base_pixel_number
        final_h = round(ratio * img_h) // base_pixel_number * base_pixel_number
    final_img = img.resize((final_w, final_h), Image.BILINEAR)

    resize_ratio = float(final_h) / img.size[1]

    # resize crop img
    crop_list = []
    tmp_resize_character_width_list = []
    # tmp_resize_character_height_list = []
    for i, it in enumerate(tmp_crop_list):
        crop_h, crop_w = tmp_character_height_list[i], tmp_character_width_list[i]
        resize_crop_h, resize_crop_w = int(crop_h * resize_ratio // base_pixel_number * base_pixel_number), int(
            crop_w * resize_ratio // base_pixel_number * base_pixel_number)
        tmp_resize_character_width_list.append(resize_crop_w)
        # tmp_resize_character_height_list.append(resize_crop_h)
        crop_list.append(cv2.resize(it, (resize_crop_w, resize_crop_h)))

    # slice final image
    slice_final_list = []
    end_x = int(offset_x * resize_ratio + tmp_resize_character_width_list[0] * (
            1 + word_space_ratio_CN / 2)) // base_pixel_number * base_pixel_number
    slice_final_list.append(final_img.crop((0, 0, end_x, final_h)))
    start_x = end_x
    for i in range(1, len(crop_list) - 1):
        end_x = int(start_x + tmp_resize_character_width_list[i - 1] * word_space_ratio_CN / 2 +
                    tmp_resize_character_width_list[i] * (
                            1 + word_space_ratio_CN / 2)) // base_pixel_number * base_pixel_number
        slice_final_list.append(final_img.crop((start_x, 0, end_x, final_h)))
        start_x = end_x
    slice_final_list.append(final_img.crop((start_x, 0, final_w, final_h)))

    # resize offset
    resize_offset_x = int(offset_x * resize_ratio // base_pixel_number * base_pixel_number)

    # slice views (h_start, h_end, w_start, w_end)
    slice_view_list = []
    end_x = int(resize_offset_x + tmp_resize_character_width_list[0] * (
            1 + word_space_ratio_CN )) // base_pixel_number * base_pixel_number

    slice_view_list.append((0, final_h // base_pixel_number, 0, end_x // base_pixel_number))
    start_x = end_x - int(word_space_ratio_CN  * tmp_resize_character_width_list[0])
    for i in range(1, len(crop_list) - 1):
        end_x = int(start_x + tmp_resize_character_width_list[i - 1] * word_space_ratio_CN +
                    tmp_resize_character_width_list[i] * (
                            1 + word_space_ratio_CN )) // base_pixel_number * base_pixel_number
        slice_view_list.append(
            (0, final_h // base_pixel_number, start_x // base_pixel_number, end_x // base_pixel_number))
        start_x = end_x - int(word_space_ratio_CN  * tmp_resize_character_width_list[i])
    slice_view_list.append(
        (0, final_h // base_pixel_number, start_x // base_pixel_number, final_w // base_pixel_number))

    # slice views according to patch (h_start, h_end, w_start, w_end)
    # slice_view_list = []
    # slice_num = len(slice_final_list)
    # if slice_num == 2:
    #     # total only 1 patch
    #     slice_view_list.append((0, final_h // base_pixel_number, 0, final_w // base_pixel_number))
    # elif slice_num == 3:
    #     # total 2 patch
    #     # 1st patch
    #     end_x = resize_offset_x + tmp_resize_character_width_list[0] * (
    #             1 + word_space_ratio_CN) + tmp_resize_character_width_list[1] * (
    #                     1 + word_space_ratio_CN)
    #     end_x = int(end_x) // base_pixel_number * base_pixel_number
    #     slice_view_list.append((0, final_h // base_pixel_number, 0, end_x // base_pixel_number))
    #
    #     # 2nd patch
    #     start_x = end_x - tmp_resize_character_width_list[1] * (
    #             1 + word_space_ratio_CN) - tmp_resize_character_width_list[0] * word_space_ratio_CN
    #     start_x = int(start_x) // base_pixel_number * base_pixel_number
    #     slice_view_list.append(
    #         (0, final_h // base_pixel_number, start_x // base_pixel_number, final_w // base_pixel_number))
    #
    #
    # elif slice_num == 4:
    #     # 1st patch
    #     end_x = resize_offset_x + tmp_resize_character_width_list[0] * (
    #             1 + word_space_ratio_CN) + tmp_resize_character_width_list[1] * (
    #                     1 + word_space_ratio_CN)
    #     end_x = int(end_x) // base_pixel_number * base_pixel_number
    #     slice_view_list.append((0, final_h // base_pixel_number, 0, end_x // base_pixel_number))
    #
    #     # 2nd patch
    #     start_x = end_x - tmp_resize_character_width_list[1] * (
    #             1 + word_space_ratio_CN) - tmp_resize_character_width_list[0] * word_space_ratio_CN
    #     start_x = int(start_x) // base_pixel_number * base_pixel_number
    #     end_x = start_x + tmp_resize_character_width_list[0] * word_space_ratio_CN + tmp_resize_character_width_list[
    #         1] * (1 + word_space_ratio_CN) + \
    #             tmp_resize_character_width_list[2] * (1 + word_space_ratio_CN)
    #     end_x = int(end_x) // base_pixel_number * base_pixel_number
    #     slice_view_list.append(
    #         (0, final_h // base_pixel_number, start_x // base_pixel_number, end_x // base_pixel_number))
    #
    #     # 3rd patch
    #     start_x = end_x - tmp_resize_character_width_list[2] * (1 + word_space_ratio_CN) - \
    #               tmp_resize_character_width_list[1] * word_space_ratio_CN
    #     start_x = int(start_x) // base_pixel_number * base_pixel_number
    #     slice_view_list.append(
    #         (0, final_h // base_pixel_number, start_x // base_pixel_number, final_w // base_pixel_number))

    return final_img, slice_final_list, slice_view_list, crop_list, text_ratio_x, text_ratio_y, resize_offset_x


def enhance_character(text, ori_img, font_path, default_font_path_list, fg_color, bg_color, alpha_ratio, font_size=812,
                      text_ratio=0.85, b_bg=False):
    # early return
    if text_ratio == 0:  # EN_char
        return ori_img

    select_font_path = obtain_draw_font(text, font_path, default_font_path_list)
    if select_font_path is None:
        return None

    # draw binary_grey img
    font = ImageFont.truetype(select_font_path, font_size)
    pad_grey, crop_grey = gen_img_grey(text, font, text_ratio)

    # create PIL Image with the same size of crop_grey
    crop_h, crop_w = crop_grey.shape

    # b_comples b_bg
    if b_bg:
        text_bg_img = Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0))  # RGBA
        draw = ImageDraw.Draw(text_bg_img)
        # modify color
        bg_color_list = list(bg_color)
        bg_color_list.append(int(255 * alpha_ratio))
        bg_color = tuple(bg_color_list)
        contours, _ = cv2.findContours(255 - crop_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            epsilon = 0.01 * cv2.arcLength(cont, True)
            box = cv2.approxPolyDP(cont, epsilon=epsilon, closed=True)
            draw.polygon(polybox2points(box), fill=bg_color)

        # convert to nparray
        text_bg_img_array = np.array(text_bg_img)
        # add text fill color
        fg_color_list = list(fg_color)
        fg_color_list.append(int(255 * alpha_ratio))
        fg_color = tuple(fg_color_list)
        crop_rgba = fill_color(crop_grey, None, fg_color, text_bg_img_array)
    else:
        # fill color and gen rgba image, keep fg color
        crop_color = fill_color(crop_grey, None, fg_color)
        crop_rgba = np.concatenate((crop_color, np.expand_dims((255 - crop_grey) * alpha_ratio, axis=2)), axis=2)

    # convert crop_rgba to PIL.Image
    crop_image = Image.fromarray(crop_rgba.astype(np.uint8))

    crop_w, crop_h = crop_image.size
    pad_h, pad_w = pad_grey.shape

    # compute resize size
    img_w, img_h = ori_img.size
    resize_ratio = float(img_w) / pad_w
    assert resize_ratio == float(img_h) / pad_h, "resize size error"

    crop_resize_w = int(crop_w * resize_ratio)
    crop_resize_h = int(crop_h * resize_ratio)

    crop_image = crop_image.resize((crop_resize_w, crop_resize_h), Image.BILINEAR)
    start_x = int((img_w - crop_resize_w) / 2)
    start_y = int((img_h - crop_resize_h) / 2)

    r, g, b, a = crop_image.split()

    ori_img = ori_img.convert("RGBA")
    ori_img.paste(crop_image, (start_x, start_y, start_x + crop_resize_w, start_y + crop_resize_h), mask=a)

    return ori_img.convert("RGB"), crop_image.convert("RGB")
