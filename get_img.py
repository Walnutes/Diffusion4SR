import os

import cv2
import numpy as np
from PIL import Image


def resizeimage(img, h=128, w=512):
    img_w, img_h = img.size
    if img_w / img_h < 1.:
        img = img.resize((h, h), Image.BILINEAR)
        resize_img = Image.new(mode='RGB', size=(w,h), color=0)
        # np.zeros((h, w, 3), dtype = np.uint8)
        # img = np.array(img, dtype = np.uint8)  # (w,h) -> (h,w,c)
        resize_img.paste(img, box=(0, 0))
        # resize_img[0:h, 0:h, :] = img
        img = resize_img

    elif img_w / img_h < w / h:
        ratio = img_h / h
        new_w = int(img_w / ratio)
        img = img.resize((new_w, h), Image.BILINEAR)
        resize_img = Image.new(mode='RGB', size=(w,h), color=0)
        # img = np.array(img, dtype = np.uint8)  # (w,h) -> (h,w,c)
        # resize_img[0:h, 0:new_w, :] = img
        resize_img.paste(img, box=(0, 0))
        img = resize_img

    else:
        img = img.resize((w, h), Image.BILINEAR)
        # resize_img = np.zeros((h, w, 3), dtype = np.uint8)
        resize_img = Image.new(mode='RGB', size=(w,h), color=0)
        # img = np.array(img, dtype = np.uint8)  # (w,h) -> (h,w,c)
        #pdb.set_trace()

        # resize_img[:, :, :] = img
        resize_img.paste(img, box=(0, 0))
        img = resize_img

    
    # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)  
    # img = img.astype(np.float32) / 255.
    return img


def pad_image(img):
    w,h = img.size
    if w>=h:
        im_new = Image.new(mode='RGB', size=(w,w), color=0)
        im_new.paste(img, box=(0, w-h))
    else:
        im_new = Image.new(mode='RGB', size=(h,h), color=0)
        im_new.paste(img, box=((h-w)//2, 0))
   
    return im_new

def pad_gt(img):
    w,h = img.size
    if h>=32:
        h_new = 32
    else:
        h_new = h
    if w>=128:
        w_new = 128
    else:
        w_new = w
    img_new = img.resize((w_new,h_new))
    pad = Image.new(mode='RGB', size=(128,32), color=0)
    pad.paste(img_new,box=(0,0))
    return pad

def test():
    # '../sr3/datasets/mosaic_val/lr_(32,128)'
    dir_tagH = "./datasets/mosaic_train/hr_(32,128)/3.jpg"

    img = Image.open(dir_tagH)
    # img = resizeimage(img)

    # sizes = img.size
    # print(sizes)

    # print(type(sizes))
    # str = '(128,64)'
    # tup = eval(str)
    # print(tup)
    # print(type(tup))
    # img = img.resize(tup)
    # print(img.size)

    # img = img.resize(sizes)
    # cv2.imwrite('./datasets/mosaic_train/examples/new/trainH_pad_0.jpg', img)
    img.save('./datasets/mosaic_train/examples/new/trainH_pad_0.jpg')

def sets(dir_orgH, dir_orgL, dir_tagH, dir_tagL, dir_tagS):
    
    for folder in os.listdir(dir_orgH):
        # print(folder)  # 测试
        for filename in os.listdir(dir_orgH + "/" + folder):
                # if filename == folder + '_330000.png':
            # print(filename)  # 测试
            img = Image.open(dir_orgH + "/" + folder + "/" + filename)
            img = resizeimage(img)
            img.save(dir_tagH+ "/" + filename)
            # img.save(dir_copy+ "/" + filename)
            
    # # for folder in os.listdir(dir_orgL):
    #     # print(folder)  # 测试
    #     for filename in os.listdir(dir_orgL):
    #             # if filename == folder + '_330000.png':
    #         # print(filename)  # 测试
    #         img = Image.open(dir_orgL  + "/" + filename)
    #         img = resizeimage(img)
    #         # img.save(dir_tagL+ "/" + filename)
    #         img.save(dir_tagS+ "/" + filename)
    #         # img.save(dir_copy+ "/" + filename)

dir_orgH = '../trainsets/trainH/images'
dir_orgL = './datasets/mosaic_train/sr_(32,128)_(32,128)'
# dir_orgH = '../testsets/testH'
# dir_orgL = './datasets/mosaic_val/sr_(32,128)_(32,128)'

dir_tagH = './datasets/mosaic_new/hr_(128,512)'
dir_tagL = './datasets/mosaic_new/lr_(32,128)'
dir_tagS = './datasets/mosaic_new/sr_(32,128)_(128,512)'
sets(dir_orgH, dir_orgL, dir_tagH, dir_tagL, dir_tagS)

# dir_orgH = '../testsets/testH'
# dir_tagH = './datasets/mosaic_val/hr_(32,128)'
# for folder in os.listdir(dir_orgH):
#         # print(folder)  # 测试
#         for filename in os.listdir(dir_orgH + "/" + folder):
#                 # if filename == folder + '_330000.png':
#             # print(filename)  # 测试
#             img = Image.open(dir_orgH + "/" + folder + "/" + filename)
#             img = resizeimage(img)
#             img.save(dir_tagH+ "/" + filename)

