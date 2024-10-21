import PIL.Image as Image

bg_img = Image.open("./tmp_example_img/lighting/bg_indoor.png")
fg_img = Image.open("./tmp_example_img/lighting/fg_cat.png")

resize_fg = fg_img.resize((512,512))
bg_img.paste(resize_fg,(256,512),resize_fg)
new_img = Image.new("RGBA",fg_img.size,(255,255,255,0))
new_img.paste(resize_fg,(256,512),resize_fg)
r,g,b,a = new_img.split()
import pdb
pdb.set_trace()