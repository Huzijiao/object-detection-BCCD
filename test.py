import cv2 as cv
import numpy as np
import joblib
from SVM import HoG
from SVM import PCA
import pandas as pd
import os

class TestDataSet(object):
    def __init__(self, root,sonroot):
        self.root = root  # 根目录
        self.sonroot = sonroot
    def datasets(self):
        # 读取每类图片的名字,获得文件夹下面所有文件的列表,文件夹为BloodImage_00000-30.jpg
        image = list(sorted(os.listdir(os.path.join(self.root, self.sonroot))))
        # 储存图片的numpy矩阵
        image_numpy = np.zeros((len(image), 256, 256), dtype=np.uint8)
        # 读取图片，进行resize至256 * 256
        for i in range(len(image)):
            img = cv.imread(os.path.join(self.root, self.sonroot, image[i]), cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (256, 256), cv.INTER_CUBIC)
            image_numpy[i] = img
        # 返回一个数组，每行代表一个图像
        return image_numpy


# 加载测试数据
df = []
# 测试30张图片
for j in range(10):
    dataset_test = TestDataSet(root='C:\\Users\\13521\\Desktop\\sxzt\\4\\object-detection-BCCD\\data\\crop_ss_\\', sonroot='BloodImage_0000'+str(j)+".jpg")
    test_data = dataset_test.datasets()
    print("查看测试数据集格式：")
    print(test_data.shape)

    # 测试集HoG特征获取
    test_feature = []
    for i in range(len(test_data)):
        hog = HoG(test_data[i], 32, 9)
        temp_feature = hog.Block_Vector()
        test_feature.append(temp_feature)
    test_feature = np.array(test_feature)
    print("测试集HoG特征获取")
    print(test_feature.shape)

    # PCA降维测试集的特征，n_components为特征向量的个数
    pca = PCA(n_components=300)
    pca.fit(test_feature)
    test_reduction = pca.transform(test_feature)
    print("特征向量PCA降维")

    # 加载训练好的Model
    myclf = joblib.load('data/save_model/model_4.pkl')
    # 使用模型进行预测
    predict_test_y = myclf.predict(test_reduction.astype(float))
    # 计算置信概率矩阵
    probs = myclf.predict_proba(test_reduction)

    print("置信概率矩阵:"+str(probs))
    print("预测向量为："+str(predict_test_y))
    # 将置信概率矩阵的最大值取出，作为分类可信度
    confi = []
    for i in range(probs.shape[0]):
        max = probs[i][0]
        if probs[i][1] > max:
            max = probs[i][1]
        if probs[i][2] > max:
            max = probs[i][2]
        confi.append(max)
    print("分类可信度为：")
    print(confi)

    # 遍历列表保存数据
    row = []
    for i in range(len(predict_test_y)):
        cell_type = predict_test_y[i]
        confidence = confi[i]
        row = [cell_type, confidence]
        df.append(row)


data = pd.DataFrame(df, columns=['cell_type', 'confidence'])
data[['cell_type', 'confidence']].to_csv('data/type_confidence.csv', index=False)