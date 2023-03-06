from keras.preprocessing import image
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import os
import pandas as pd
import csv


df = []
# 读取图片的名字,获得文件夹下面所有文件的列表
all_image_list = list(sorted(os.listdir(os.path.join('data', 'JPEGImages'))))
for name in all_image_list:
    img_path = os.path.join('data', 'JPEGImages', name)
    # v2.useOptimized()函数可以查看当前优化是否开启,cv2.setUseOptimized()可以设置是否开启优化
    # 如果报错，检查是否安装的是opencv，如果安装的opencv(简版不包含一些函数)，卸载安装opencv-contrib-python
    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    #读入图片
    im = cv2.imread(img_path)
    ss.setBaseImage(im)
    #利用快速选择性搜索
    ss.switchToSelectiveSearchFast()
    #提取的候选框,rects保存了候选框的边框坐标
    rects = ss.process()
    print(rects.shape)
    imOut = im.copy()

    # 绘制候选框
    # plt.imshow(im)
    # for i, rect in (enumerate(rects)):
    #     x, y, w, h = rect
    #     print(x,y,w,h)
    # #   imOut = imOut[x:x+w,y:y+h]
    #     cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    # # plt.figure()
    # plt.imshow(imOut)
    # # plt.show()
    row = []

    # 根据坐标信息裁剪图像
    for i, rect in (enumerate(rects)):
        x, y, w, h = rect
        x1 = x+w
        if x+w > 639:
            x1 = 639
        y1 = y+h
        if y1 > 479:
            y1 = 479
        # 如果子图尺寸太大或太小，跳过本次循环
        if (x1 - x + 1) * (y1 - y + 1) > 82144+10000 or (x1 - x + 1) * (y1 - y + 1) < 480-150:
            continue
        crop_img = imOut[y:y1, x:x1]
        # crop_img_name的命名方式：img_name+"_"+label+"_"+i
        crop_img_name = str(i) + ".jpg"
        crop_img_save_dir = os.path.join(os.getcwd(), "data/crop_ss_", name)
        print(crop_img_save_dir)
        # 不存在则创建路径
        if os.path.exists(crop_img_save_dir) == 0:
            os.makedirs(crop_img_save_dir)
        crop_img_save_path = os.path.join(crop_img_save_dir, crop_img_name)
        cv2.imwrite(crop_img_save_path, crop_img)
        # 将坐标信息保存到csv文件中
        xmin = x
        xmax = x1
        ymin = y
        ymax = y1
        filename = name
        row = [filename, xmin, xmax, ymin, ymax]
        df.append(row)
        print(name)
    if filename=='BloodImage_00009.jpg':
        break


data = pd.DataFrame(df, columns=['filename','xmin', 'xmax', 'ymin', 'ymax'])

data[['filename', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('ss_for_nms.csv', index=False)
