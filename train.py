import cv2 as cv
import numpy as np
import joblib
from sklearn import svm

from SVM import HoG, micro_f1, DataSet
from SVM import PCA
import pandas as pd
import os

#加载数据用于训练模型
dataset = DataSet(root='data/crop_img', division=0.875)
train_data, train_target, test_data, test_target = dataset.datasets()
print("查看数据集格式：")
print(train_data.shape, train_target.shape, test_data.shape, test_target.shape)
print("查看数据集目标值：")
print(train_target, test_target)
#
# # 加载测试数据
# dataset_test = TestDataSet(root='data/crop_from_ss')
# test_data = dataset_test.datasets()
# print("查看测试数据集格式：")
# print(test_data.shape)
#
#
# 训练集HOG特征获取
train_feature = []
for i in range(len(train_data)):
    hog = HoG(train_data[i], 32, 9)
    temp_feature = hog.Block_Vector()
    train_feature.append(temp_feature)
train_feature = np.array(train_feature)
print("查看训练集特征的格式：")
print(train_feature.shape)
#
# 测试集HOG特征获取
test_feature = []
for i in range(len(test_data)):
    hog = HoG(test_data[i], 32, 9)
    temp_feature = hog.Block_Vector()
    test_feature.append(temp_feature)
test_feature = np.array(test_feature)
print("查看测试集特征的格式：")
print(test_feature.shape)
#
# 对训练数据PCA降维，n_components为主成分的个数
pca_train = PCA(n_components = 300)
pca_train.fit(train_feature)
train_reduction = pca_train.transform(train_feature)
print("查看降维后的训练集格式：")
print(train_reduction.shape)
#
# 对测试数据PCA降维
pca_test = PCA(n_components = 300)
pca_test.fit(test_feature)
test_reduction = pca_test.transform(test_feature)
print("查看降维后的测试集格式：")
print(test_reduction.shape)
#
# 模型训练和保存,调用C-SVM分类器,预测的时候注释
'''C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’,
					coef0=0.0, shrinking=True, probability=False, tol=0.001,
					cache_size=200, class_weight=None, verbose=False, max_iter=-1,
					decision_function_shape=’ovr’, random_state=None
'''
clf = svm.SVC(C=0.7,kernel='rbf', probability=True)
clf.fit(train_reduction, train_target)
print(clf.score(train_reduction, train_target))
joblib.dump(clf, 'data/save_model/model_8.pkl')
#
# 加载训练好的Model
print("加载模型")
clf = joblib.load('data/save_model/model_8.pkl')
# # 使用模型进行预测
predict_test_y = clf.predict(test_reduction.astype(float))
# # 计算置信概率矩阵
# probs = clf.predict_proba(test_reduction)
# print("置信概率矩阵:")
# print(probs)
# print("预测向量为："+str(predict_test_y))
# # 将置信概率矩阵的最大值取出，作为分类可信度
# confidence = []
# for i in range(probs.shape[0]):
#     max = probs[i][0]
#     if probs[i][1] > max:
#         max = probs[i][1]
#     if probs[i][2] > max:
#         max = probs[i][2]
#     confidence.append(max)
# print("分类可信度为：")
# print(confidence)
# 将每一类


precision, recall, f1, TP1, TP2, TP3= micro_f1(predict_test_y, test_target)
print("精确率，召回率，f1值，多分类问题中，它们的值都等于精确率（Accuracy）")
print(precision, recall, f1)
print("TP1个数, TP2个数, TP3个数:")
print(TP1, TP2, TP3)
# print("血小板，红细胞，白细胞的分类精确度分别为：")
# print(precision1, precision2, precision3)
print(clf.score(test_reduction, test_target))