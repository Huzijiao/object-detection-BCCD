"""
根据xml文件的类别信息和位置坐标信息，将对应的类别对象裁剪出来，并存入到以类别信息命名的文件夹中
输入：包含图片的文件夹imgs，包含xml信息的文件夹xmls
输出：文件夹crop_img，crop_img下的以标签类别命名的子文件夹，以及crop的子图
"""
import cv2
import os
import xml.etree.ElementTree as ET
xml_file_dir = "data/trainval_Annotations"
img_file_dir = "data/trainval_JPEGImages"
xmls = os.listdir(xml_file_dir)
for xml in xmls:
    # 解析xml
    tree = ET.parse(os.path.join(xml_file_dir,xml))
    root = tree.getroot()
    # 找到图像名称
    img_name = root.find("filename").text
    # 读取图像
    img = cv2.imread(os.path.join(img_file_dir,img_name))
    # 找到标注对象
    objects = root.findall('object')
    for i, obj in enumerate(objects):
        id = 0
        label = obj.find('name').text
        if label == ":":
            label = "colon"
        bb = obj.find('bndbox')
        xmin = bb.find('xmin').text
        if int(xmin)<0:
            xmin=0
        ymin = bb.find('ymin').text
        if int(ymin) < 0:
            ymin = 0
        xmax = bb.find('xmax').text
        # if int(xmax) > 639:
        #     xmax = 639
        ymax = bb.find('ymax').text
        # if int(ymax) < 439:
        #     ymax = 439

        # 保存crop img
        crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

        # crop_img_name的命名方式：img_name+"_"+label+"_"+i
        crop_img_name = os.path.splitext(img_name)[0] + "_" + label+"_"+str(i)+".jpg"

        crop_img_save_dir = os.path.join(os.getcwd(),"data/crop_img",label)
        print(crop_img_save_dir)
        if os.path.exists(crop_img_save_dir) == 0:
            os.makedirs(crop_img_save_dir)
        crop_img_save_path = os.path.join(crop_img_save_dir, crop_img_name)
        cv2.imwrite(crop_img_save_path,crop_img)

'''代码修改自csdn'''
'''	cv2.imwrite()，首先确保路径正确，其次img注意以下问题：
	1.分割的最小值不能小于0
	2.分割的最大值不能大于图片的宽和高
	3.分割的最大值要大于分割的最小值
'''