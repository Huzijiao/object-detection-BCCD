import numpy as np
import random
import cv2
import pandas as pd



def non_max_suppress(predicts_dict, threshold):

    for object_name, bbox in predicts_dict.items():  # 对每一个类别分别进行NMS；一次读取一对键值（即某个类别的所有框）
        bbox_array = np.array(bbox, dtype=np.float)
        # 下面分别获取框的左上角坐标（x1，y1），右下角坐标（x2，y2）及此框的置信度
        x1 = bbox_array[:, 0]  # 取出第0列
        x2 = bbox_array[:, 1]
        y1 = bbox_array[:, 2]
        y2 = bbox_array[:, 3]
        scores = bbox_array[:, 4]
        order = scores.argsort()[::-1]  # argsort函数返回的是数组值从小到大的索引值,[::-1]表示取反。即这里返回的是数组值从大到小的索引值
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 当前类所有框的面积(python会自动使用广播机制，相当于MATLAB中的.*即两矩阵对应元素相乘)
        keep = []

        # 按confidence从高到低遍历bbx，移除所有与该矩形框的IoU值大于threshold的矩形框
        while order.size > 0:
            i = order[0]
            keep.append(i)  # 保留当前最大confidence对应的bbx索引
            # 获取所有与当前bbx的交集对应的左上角和右下角坐标，并计算IoU（注意这里是同时计算一个bbx与其他所有bbx的IoU）

            xx1 = np.maximum(x1[i], x1[order[1:]])  # 最大置信度的左上角坐标分别与剩余所有的框的左上角坐标进行比较，分别保存较大值；因此这里的xx1的维数应该是当前类的框的个数减1
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)  # 注意这里都是采用广播机制，同时计算了置信度最高的框与其余框的IoU
            inds = np.where(iou <= threshold)[0]  # 保留iou小于等于阙值的框的索引值
            order = order[inds + 1]  # 将order中的第inds+1处的值重新赋值给order；即更新保留下来的索引，加1是因为因为没有计算与自身的IOU，所以索引相差１，需要加上
        bbox = bbox_array[keep]
        predicts_dict[object_name] = bbox.tolist()
    # predicts_dict = predicts_dict
    return predicts_dict


df1 = []
for j in range(10):
    name = 'BloodImage_0000' + str(j) + '.jpg'
    print(name)
    # 取出字典需要的值,1是血小板类
    df = pd.read_csv('ss_for_nms.csv')
    value1 = []
    for i, line in df.iterrows():
        l1 = []
        area1 = (line['xmax'] - line['xmin']) * (line["ymax"] - line["ymin"])
        conf1 = line["confidence"]
        if line['filename']==name and line['cell_type'] == 1 and area1>600 and area1<1600 and line['confidence']>0.9 :
            l1.append(line['xmin'])
            l1.append(line['xmax'])
            l1.append(line['ymin'])
            l1.append(line['ymax'])
            l1.append(line['confidence'])
            value1.append(l1)
            print(l1)
        if i > 13217:
            break


    # 2为红细胞3000-10000
    value2 = []
    for i, line in df.iterrows():
        l2 = []
        area2 = (line['xmax'] - line['xmin']) * (line["ymax"] - line["ymin"])
        if line['filename']==name and line['cell_type'] == 2 and area2>2500 and area2<30000:
            l2.append(line['xmin'])
            l2.append(line['xmax'])
            l2.append(line['ymin'])
            l2.append(line['ymax'])
            l2.append(line['confidence'])
            value2.append(l2)
            print(l2)
        if i > 13217:
            break

    # 3为白细胞
    value3 = []
    for i, line in df.iterrows():
        l3 = []
        area3 = (line['xmax'] - line['xmin']) * (line["ymax"] - line["ymin"])
        if line['filename'] ==name and line['cell_type'] == 3 :
            l3.append(line['xmin'])
            l3.append(line['xmax'])
            l3.append(line['ymin'])
            l3.append(line['ymax'])
            l3.append(line['confidence'])
            value1.append(l3)
            print(l3)
        if i > 13217:
            break


    # 构造predicts_dict

    predicts_dict = {}
    if len(value1) != 0:
        predicts_dict[1] = value1
    if len(value2) != 0:
        predicts_dict[2] = value2
    if len(value3) != 0:
        predicts_dict[3] = value3
    print(name+"字典构造完成")
    print(predicts_dict)

    predicts_dict_nms = non_max_suppress(predicts_dict, 0.4)


    # 把生成的NMS框保存至新的nms_for_plot.csv文件中


    for object_name, bbox in predicts_dict.items():
        row = []
        for box in bbox:
            x1, x2, y1, y2 = box[0], box[1], box[2], box[3]
            row = [name,x1,x2,y1,y2,object_name]
            df1.append(row)
data = pd.DataFrame(df1, columns=['filename','xmin', 'xmax', 'ymin', 'ymax','cell_type'])
data[['filename','xmin', 'xmax', 'ymin', 'ymax','cell_type']].to_csv('data/nms_for_plot.csv', index=False)






# # 下面在一张全黑图片上测试非极大值抑制的效果
# img = np.zeros((600, 600), np.uint8)
# # predicts_dict = {'black1': [[83, 54, 165, 163, 0.8], [67, 48, 118, 132, 0.5], [91, 38, 192, 171, 0.6]]}
# # predicts_dict = {'black1': [[83, 54, 165, 163, 0.8], [67, 48, 118, 132, 0.5], [91, 38, 192, 171, 0.6]],
# #                  'black2': [[59, 120, 137, 368, 0.12], [54, 154, 148, 382, 0.13]]}
# # 在全黑的图像上画出设定的几个框
# for object_name, bbox in predicts_dict.items():
#     for box in bbox:
#         x1, y1, x2, y2, score = box[0], box[1], box[2], box[3], box[-1]
#         y_text = int(random.uniform(y1,
#                                     y2))  # uniform()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法。uniform() 方法将随机生成下一个实数，它在 [x, y) 范围内
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
#         cv2.putText(img, str(score), (x2 - 30, y_text), 2, 1, (255, 255, 0))
#     cv2.namedWindow("black1_roi")  # 创建一个显示图像的窗口
#     cv2.imshow("black1_roi", img)  # 在窗口中显示图像;注意这里的窗口名字如果不是刚刚创建的窗口的名字则会自动创建一个新的窗口并将图像显示在这个窗口
#     cv2.waitKey(0)  # 如果不添这一句，在IDLE中执行窗口直接无响应。在命令行中执行的话，则是一闪而过。
#     cv2.destroyAllWindows()  # 最后释放窗口是个好习惯！
#
# # 在全黑图片上画出经过非极大值抑制后的框
# img_cp = np.zeros((600, 600), np.uint8)
# predicts_dict_nms = non_max_suppress(predicts_dict, 0.5)
# for object_name, bbox in predicts_dict_nms.items():
#     for box in bbox:
#         x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[-1]
#         y_text = int(random.uniform(y1,
#                                     y2))  # uniform()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法。uniform() 方法将随机生成下一个实数，它在 [x, y) 范围内
#         cv2.rectangle(img_cp, (x1, y1), (x2, y2), (255, 255, 255), 2)
#         cv2.putText(img_cp, str(score), (x2 - 30, y_text), 2, 1, (255, 255, 0))
#     cv2.namedWindow("black1_nms")  # 创建一个显示图像的窗口
#     cv2.imshow("black1_nms", img_cp)  # 在窗口中显示图像;注意这里的窗口名字如果不是刚刚创建的窗口的名字则会自动创建一个新的窗口并将图像显示在这个窗口
#     cv2.waitKey(0)  # 如果不添这一句，在IDLE中执行窗口直接无响应。在命令行中执行的话，则是一闪而过。
#     cv2.destroyAllWindows()  # 最后释放窗口是个好习惯！
'''代码修改自https://zhuanlan.zhihu.com/p/40976906'''