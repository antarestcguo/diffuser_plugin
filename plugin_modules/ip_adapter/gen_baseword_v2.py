from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from fontTools.ttLib import TTFont

default_font_path = ['./resource/font/ArialNarrowBold.ttf', './resource/font/STBaoliSC-Regular-01.ttf']
default_font_type = ["Arial", "baoli"]

max_chinese_character_num = 4


def is_chinese(char):
    # Check if a character is Chinese
    return '\u4e00' <= char <= '\u9fff'


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

def create_maxtext_image_RGB_CN(text, bg_color, fg_color, font_path, font_size, max_side=1920, min_side=768,text_ratio=0.85):
    return

def create_maxtext_image_RGB_EN(text, bg_color, fg_color, font_path, font_size, max_side=1920, min_side=768,text_ratio=0.85):
    tokens = text.strip().split(" ")
    text_to_draw = tokens[0]
    b_draw = True
    select_font_path = font_path
    for it_text in text_to_draw:
        b_draw = b_text_can_be_draw(font_path, it_text)
        if not b_draw:
            for it_default_font_path in default_font_path:
                b_draw = b_text_can_be_draw(it_default_font_path, it_text)
                if b_draw:
                    select_font_path = it_default_font_path
                    break
        if not b_draw:
            return None,None
    font = ImageFont.truetype(select_font_path, font_size)
    text_width, text_height = font.getsize(text_to_draw)

    # cv2 compute bbox
    image_text_grey = Image.new("RGB", (text_width, text_height), (0, 0, 0))  # RGB, not RGBA
    draw_grey = ImageDraw.Draw(image_text_grey)
    draw_grey.text((0, 0), text_to_draw, font=font, fill=(255, 255, 255))
    # convert 2 opencv , compute the min bbox
    cv_img = cv2.cvtColor(np.asarray(image_text_grey), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(cv_img, 127, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(binary)

    # img text
    image_text = Image.new("RGB", (text_width, text_height), bg_color)  # RGB, not RGBA
    draw = ImageDraw.Draw(image_text)
    draw.text((0, 0), text_to_draw, font=font, fill=fg_color)

    # merge img_bg and img_text
    image_text = image_text.crop((x, y, x + w, y + h))
    text_w, text_h = image_text.size

    # compute bg_img w and h
    bg_img_w = text_w / text_ratio
    bg_img_h = text_h / text_ratio

    ratio = min_side / min(bg_img_w, bg_img_h)
    bg_img_w = bg_img_w * ratio
    bg_img_h = bg_img_h * ratio

    ratio = max_side / max(bg_img_h, bg_img_w)
    bg_img_w = bg_img_w * ratio
    bg_img_h = bg_img_h * ratio

    # bg image
    image_bg = Image.new("RGB", (int(bg_img_w), int(bg_img_h)), bg_color)  # RGB, not RGBA

    # resize the text image
    new_text_w = int(bg_img_w * text_ratio)
    new_text_h = int(bg_img_h * text_ratio)
    resize_image_text = image_text.resize((new_text_w, new_text_h), Image.ANTIALIAS)

    # re-compute the w and h
    text_w, text_h = resize_image_text.size

    start_x = int((bg_img_w - text_w) / 2)
    start_y = int((bg_img_h - text_h) / 2)
    image_bg.paste(resize_image_text, (start_x, start_y, start_x + text_w, start_y + text_h))


    return image_bg,select_font_path

def create_maxtext_image_RGB(text, bg_color, fg_color, font_path, font_size, max_side=1920, min_side=768,
                             text_ratio=0.85):
    text_to_draw = text[:max_chinese_character_num]

    # check if can be draw
    b_draw = True
    select_font_path = font_path
    for it_text in text_to_draw:
        b_draw = b_text_can_be_draw(font_path, it_text)
        if not b_draw:
            for it_default_font_path in default_font_path:
                b_draw = b_text_can_be_draw(it_default_font_path, it_text)
                if b_draw:
                    select_font_path = it_default_font_path
                    break
        if not b_draw:
            return None
    font = ImageFont.truetype(select_font_path, font_size)
    text_width, text_height = font.getsize(text_to_draw)

    # cv2 compute bbox
    image_text_grey = Image.new("RGB", (text_width, text_height), (0, 0, 0))  # RGB, not RGBA
    draw_grey = ImageDraw.Draw(image_text_grey)
    draw_grey.text((0, 0), text_to_draw, font=font, fill=(255, 255, 255))
    # convert 2 opencv , compute the min bbox
    cv_img = cv2.cvtColor(np.asarray(image_text_grey), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(cv_img, 127, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(binary)

    # img text
    image_text = Image.new("RGB", (text_width, text_height), bg_color)  # RGB, not RGBA
    draw = ImageDraw.Draw(image_text)
    draw.text((0, 0), text_to_draw, font=font, fill=fg_color)

    # merge img_bg and img_text
    image_text = image_text.crop((x, y, x + w, y + h))
    text_w, text_h = image_text.size

    # compute bg_img w and h
    bg_img_w = text_w / text_ratio
    bg_img_h = text_h / text_ratio

    ratio = min_side / min(bg_img_w, bg_img_h)
    bg_img_w = bg_img_w * ratio
    bg_img_h = bg_img_h * ratio

    ratio = max_side / max(bg_img_h, bg_img_w)
    bg_img_w = bg_img_w * ratio
    bg_img_h = bg_img_h * ratio

    # bg image
    image_bg = Image.new("RGB", (int(bg_img_w), int(bg_img_h)), bg_color)  # RGB, not RGBA

    # resize the text image
    new_text_w = int(bg_img_w * text_ratio)
    new_text_h = int(bg_img_h * text_ratio)
    resize_image_text = image_text.resize((new_text_w, new_text_h), Image.ANTIALIAS)

    # re-compute the w and h
    text_w, text_h = resize_image_text.size

    start_x = int((bg_img_w - text_w) / 2)
    start_y = int((bg_img_h - text_h) / 2)
    image_bg.paste(resize_image_text, (start_x, start_y, start_x + text_w, start_y + text_h))

    return image_bg


def polybox2points(box):
    points = []
    for it in box:
        points.append(tuple(it[0]))

    return points


def paste_maxtext_EN(text, ori_image, fg_color, bg_color, font_path, font_size, alpha_ratio, text_ratio=0.85):
    tokens = text.strip().split(" ")
    text_to_draw = tokens[0]

    img_w, img_h = ori_image.size

    # modify color
    fg_color_list = list(fg_color)
    fg_color_list.append(int(255 * alpha_ratio))
    fg_color = tuple(fg_color_list)

    bg_color_list = list(bg_color)
    bg_color_list.append(int(255 * alpha_ratio))
    bg_color = tuple(bg_color_list)

    font = ImageFont.truetype(font_path, font_size)

    # cv2 compute bbox
    image_text_grey = Image.new("RGB", (img_w, img_h), (0, 0, 0))  # RGB, not RGBA
    draw_grey = ImageDraw.Draw(image_text_grey)
    draw_grey.text((0, 0), text_to_draw, font=font, fill=(255, 255, 255))
    # convert 2 opencv , compute the min bbox
    cv_img = cv2.cvtColor(np.asarray(image_text_grey), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(cv_img, 20, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(binary)

    # draw text
    image_text = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))  # RGBA
    draw = ImageDraw.Draw(image_text)

    # compute polylines
    # if b_bg:
    #     contours, _ = cv2.findContours(cv_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     for cont in contours:
    #         epsilon = 0.01 * cv2.arcLength(cont, True)
    #         box = cv2.approxPolyDP(cont, epsilon=epsilon, closed=True)
    #         draw.polygon(polybox2points(box), fill=bg_color)

    draw.text((0, 0), text_to_draw, font=font, fill=fg_color)

    # merge img_bg and img_text
    image_text = image_text.crop((x, y, x + w, y + h))
    text_w, text_h = image_text.size

    # resize the image
    text_max_side = max(text_w, text_h)
    resize_max = int(img_h * text_ratio)  # default img_h == img_w
    new_text_w = int(resize_max * text_w / text_max_side)
    new_text_h = int(resize_max * text_h / text_max_side)
    resize_image_text = image_text.resize((new_text_w, new_text_h), Image.ANTIALIAS)

    # re-compute the w and h
    text_w, text_h = resize_image_text.size

    # add alpha
    r, g, b, a = resize_image_text.split()

    start_x = int((img_w - text_w) / 2)
    start_y = int((img_h - text_h) / 2)
    ori_image = ori_image.convert("RGBA")
    ori_image.paste(resize_image_text, (start_x, start_y, start_x + text_w, start_y + text_h), mask=a)

    return ori_image.convert("RGB"), resize_image_text.convert("RGB")


def compute_complex_word(text, font_path, font_size, img_w=1024, img_h=1024):
    # Determine text to draw based on input language
    if is_chinese(text[0]):
        text_to_draw = text[0]
        # Load font
        font = ImageFont.truetype(font_path, font_size)
    else:
        text_to_draw = text.split()[0]
        # Load font
        font = ImageFont.truetype(font_path, font_size)

    image_text = Image.new("RGBA", (img_w, img_h), (0, 0, 0))  # RGB
    draw = ImageDraw.Draw(image_text)
    draw.text((0, 0), text_to_draw, font=font, fill=(255, 255, 255))

    # cv2 compute bbox
    image_text_grey = Image.new("RGB", (img_w, img_h), (0, 0, 0))  # RGB, not RGBA
    draw_grey = ImageDraw.Draw(image_text_grey)
    draw_grey.text((0, 0), text_to_draw, font=font, fill=(255, 255, 255))

    # convert 2 opencv , compute the min bbox
    cv_img = cv2.cvtColor(np.asarray(image_text_grey), cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(cv_img)
    crop_grey = cv_img[y:y + h, x:x + w]

    # compute white:text, bg:black ratio
    white_idx = len(np.where(crop_grey > 127)[0])
    black_idx = len(np.where(crop_grey < 127)[0])

    new_h, new_w = crop_grey.shape

    return float(white_idx) / (new_h * new_w), float(black_idx) / (new_h * new_w)


def image_draw_color(ori_image, fg_color, bg_color, max_side=1920, min_side=768):
    # ori_image: PIL.Image, bg=(255,255,255),fg=(0,0,0)
    cv_img = cv2.cvtColor(np.asarray(ori_image), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(cv_img, 20, 255, cv2.THRESH_BINARY)

    # fill fg/bg color
    h, w = cv_img.shape
    color_img = np.zeros([h, w, 3],dtype=np.uint8)
    color_img[(binary == 255), :] = bg_color
    color_img[(binary == 0), :] = fg_color

    # convert to PIL Image
    draw_img = Image.fromarray(color_img)

    # resize according to max-side min-side
    # resize define
    base_pixel_number = 8
    mode = Image.BILINEAR

    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    draw_img = draw_img.resize([w_resize_new, h_resize_new], mode)

    return draw_img
