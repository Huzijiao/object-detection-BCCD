import cv2 as cv
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

class HoG(object):
    def __init__(self, img, cell_w, bin_count):
        rows, cols = img.shape
        img = np.power(img / 255.0, 2.2) * 255
        self.img = img
        self.cell_w = cell_w
        self.bin_count = bin_count
        self.angle_unit = 180.0 / bin_count
        self.cell_x = int(rows / cell_w)
        self.cell_y = int(cols / cell_w)

    # 求每个像素的x和y方向的梯度值和梯度方向
    def Pixel_gradient(self):
        gradient_values_x = cv.Sobel(self.img, cv.CV_64F, 1, 0, ksize=5)  # x方向梯度
        gradient_values_y = cv.Sobel(self.img, cv.CV_64F, 0, 1, ksize=5)  # y方向梯度
        gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))  # 计算总梯度
        #         gradient_angle = cv.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)#计算梯度方向
        gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)
        gradient_angle[gradient_angle > 0] *= 180 / 3.14
        gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + 3.14) * 180 / 3.14

        return gradient_magnitude, gradient_angle

    # 求每个cell的x和y方向的梯度值和梯度方向
    def Cell_gradient(self, gradient):
        cell = np.zeros((self.cell_x, self.cell_y, self.cell_w, self.cell_w))
        gradient_x = np.split(gradient, self.cell_x, axis=0)
        for i in range(self.cell_x):
            gradient_y = np.split(gradient_x[i], self.cell_y, axis=1)
            for j in range(self.cell_y):
                cell[i][j] = gradient_y[j]
        return cell

    # 对每个梯度方向进行投票
    def Get_bins(self, cell_gradient, cell_angle):
        bins = np.zeros((cell_gradient.shape[0], cell_gradient.shape[1], self.bin_count))
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                tmp_unit = np.zeros(self.bin_count)
                cell_gradient_list = np.int8(cell_gradient[i][j].flatten())
                cell_angle_list = cell_angle[i][j].flatten()
                cell_angle_list = np.int8(cell_angle_list / self.angle_unit)  # 0-9
                cell_angle_list[cell_angle_list >= 9] = 0
                #                 cell_angle_list = cell_angle_list.flatten()
                #                 cell_angle_list = np.int8(cell_angle_list / self.angle_unit) % self.bin_count

                for m in range(len(cell_angle_list)):
                    tmp_unit[cell_angle_list[m]] += int(cell_gradient_list[m])  # 将梯度值作为投影的权值

                bins[i][j] = tmp_unit
        return bins

        # 获取整幅图像的特征向量

    def Block_Vector(self):
        gradient_magnitude, gradient_angle = self.Pixel_gradient()
        cell_gradient_values = self.Cell_gradient(gradient_magnitude)
        cell_angle = self.Cell_gradient(gradient_angle)
        bins = self.Get_bins(cell_gradient_values, cell_angle)

        block_vector = []
        for i in range(self.cell_x - 1):
            for j in range(self.cell_y - 1):
                feature = []
                feature.extend(bins[i][j])
                feature.extend(bins[i + 1][j])
                feature.extend(bins[i][j + 1])
                feature.extend(bins[i + 1][j + 1])

                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(feature)
                if magnitude != 0:
                    normalize = lambda vector, magnitude: [element / magnitude for element in vector]
                    feature = normalize(feature, magnitude)

                block_vector.extend(feature)
        return np.array(block_vector)

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        def deMean(X):
            return X - np.mean(X, axis=0)

        def calcCov(X):
            return np.cov(X, rowvar=False)

        def deEigenvalue(cov):
            return np.linalg.eig(cov)

        n, self.d = X.shape
        assert self.n_components <= self.d
        # assert self.n_components <= n

        X = deMean(X)
        cov = calcCov(X)
        eigenvalue, featurevector = deEigenvalue(cov)
        index = np.argsort(eigenvalue)
        n_index = index[-self.n_components:]
        self.w = featurevector[:, n_index].real.astype(np.float32)

        return self

    def transform(self, X):
        n, d = X.shape
        assert d == self.d
        return np.dot(X, self.w)

class TestDataSet(object):
    def __init__(self, root):
        self.root = root  # 根目录
    def datasets(self):
        # 读取每类图片的名字,获得文件夹下面所有文件的列表,文件夹为BloodImage_00000.jpg
        image = list(sorted(os.listdir(os.path.join(self.root, 'BloodImage_00000.jpg'))))
        # 储存图片的numpy矩阵
        image_numpy = np.zeros((len(image), 256, 256), dtype=np.uint8)
        # 读取图片，进行resize至256 * 256
        for i in range(len(image)):
            img = cv.imread(os.path.join(self.root, 'BloodImage_00000.jpg', image[i]), cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (256, 256), cv.INTER_CUBIC)
            image_numpy[i] = img
        # 返回一个数组，每行代表一个图像
        return image_numpy

class DataSet(object):
    def __init__(self, root, division):
        self.root = root  # 根目录
        self.division = division

    def data_segmentation(self, Platelets, RBC, WBC):
        # 将每一类图片分割成训练集和测试集，四种类分别设置标签是[1, 2, 3]方便后续的性能评估
        train_Platelets, test_Platelets = Platelets[: int(Platelets.shape[0] * self.division)], Platelets[int(Platelets.shape[0] * self.division):]
        train_Platelets_target, test_Platelets_target = np.full(len(train_Platelets), 1, dtype=np.int64), np.full(len(test_Platelets), 1,
                                                                                             dtype=np.int64)
        train_RBC, test_RBC = RBC[: int(RBC.shape[0] * self.division)], RBC[int(RBC.shape[0] * self.division):]
        train_RBC_target, test_RBC_target = np.full(len(train_RBC), 2, dtype=np.int64), np.full(len(test_RBC), 2,
                                                                                                dtype=np.int64)
        train_WBC, test_WBC = WBC[: int(WBC.shape[0] * self.division)], WBC[int(WBC.shape[0] * self.division):]
        train_WBC_target, test_WBC_target = np.full(len(train_WBC), 3, dtype=np.int64), np.full(len(test_WBC), 3,
                                                                                                   dtype=np.int64)

        # 将四类图片拼接成一个大的矩阵
        train_data = np.concatenate([train_Platelets, train_RBC, train_WBC])
        test_data = np.concatenate([test_Platelets, test_RBC, test_WBC])
        train_target = np.concatenate([train_Platelets_target, train_RBC_target, train_WBC_target])
        test_target = np.concatenate([test_Platelets_target, test_RBC_target, test_WBC_target])

        # 以索引方式打乱训练集
        index = [i for i in range(len(train_data))]
        random.shuffle(index)
        train_data = train_data[index]
        train_target = train_target[index]

        return train_data, train_target, test_data, test_target

    def datasets(self):
        # 读取每类图片的名字,获得文件夹下面所有文件的列表
        image_Platelets = list(sorted(os.listdir(os.path.join(self.root, 'Platelets'))))
        image_RBC = list(sorted(os.listdir(os.path.join(self.root, 'RBC'))))
        image_WBC = list(sorted(os.listdir(os.path.join(self.root, 'WBC'))))
        #image_snake = list(sorted(os.listdir(os.path.join(self.root, 'snake'))))

        # 储存图片的numpy矩阵
        Platelets = np.zeros((len(image_Platelets), 256, 256), dtype=np.uint8)
        RBC = np.zeros((len(image_RBC), 256, 256), dtype=np.uint8)
        WBC = np.zeros((len(image_WBC), 256, 256), dtype=np.uint8)
        #snake = np.zeros((4000, 256, 256), dtype=np.uint8)

        # 读取各个类别的图片，进行resize至256 * 256
        for i in range(len(image_Platelets)):
            img = cv.imread(os.path.join(self.root, 'Platelets', image_Platelets[i]), cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (256, 256), cv.INTER_CUBIC)
            Platelets[i] = img
        for i in range(len(image_RBC)):
            img = cv.imread(os.path.join(self.root, 'RBC', image_RBC[i]), cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (256, 256), cv.INTER_CUBIC)
            RBC[i] = img
        for i in range(len(image_WBC)):
            img = cv.imread(os.path.join(self.root, 'WBC', image_WBC[i]), cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (256, 256), cv.INTER_CUBIC)
            WBC[i] = img
        print(Platelets.shape, RBC.shape, WBC.shape)
        # 分割
        train_data, train_target, test_data, test_target = self.data_segmentation(Platelets, RBC, WBC)
        return train_data, train_target, test_data, test_target

# 传入测试特征向量和测试集的真实值,计算精确率，召回率，F1值是精确率和召回率的调和均值
def micro_f1(pred, target):
    # 第一类
    target1 = target.copy()
    pred1 = pred.copy()
    target1 = target1 == 1
    pred1 = pred1 == 1
    TP1 = np.sum(target1[pred1 == 1] == 1)
    FN1 = np.sum(target1[pred1 == 0] == 1)
    FP1 = np.sum(target1[pred1 == 1] == 0)
    TN1 = np.sum(target1[pred1 == 0] == 0)

    # 第二类
    target2 = target.copy()
    pred2 = pred.copy()
    target2 = target2 == 2
    pred2 = pred2 == 2
    TP2 = np.sum(target2[pred2 == 1] == 1)
    FN2 = np.sum(target2[pred2 == 0] == 1)
    FP2 = np.sum(target2[pred2 == 1] == 0)
    TN2 = np.sum(target2[pred2 == 0] == 0)

    # 第三类
    target3 = target.copy()
    pred3 = pred.copy()
    target3 = target3 == 3
    pred3 = pred3 == 3
    TP3 = np.sum(target3[pred3 == 1] == 1)
    FN3 = np.sum(target3[pred3 == 0] == 1)
    FP3 = np.sum(target3[pred3 == 1] == 0)
    TN3 = np.sum(target3[pred3 == 0] == 0)

    TP = TP1 + TP2 + TP3
    FN = FN1 + FN2 + FN3
    FP = FP1 + FP2 + FP3
    TN = TN1 + TN2 + TN3

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    # precision1 = TP1 / (TP1 + FP1)
    # precision2 = TP2 / (TP2 + FP2)
    # precision3 = TP3 / (TP3 + FP3)
    return precision, recall, f1, TP1, TP2, TP3

