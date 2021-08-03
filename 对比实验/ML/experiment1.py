from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
# import tensorflow as tf
from sklearn.model_selection import cross_validate

import pre
import dataset
import model
import os
import random
import index
import time
from data_read import data_read
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
'''
实验1:二分类实验
存在比例划分
return:
acc
pre
rec
f1
'''
x_train = []
y_train = []
x_test = []
# 自测
y_test = []
y_label = []
X = []
Y = []
# 进行实验几
shiyan1 = True
print("begin system")
print(time.asctime(time.localtime(time.time())))
x_black, x_white = data_read()
ratio = [1, 0.1, 0.05, 0.01, 0.001]
# ratio = [1]
for sample in x_black:
    X.append(sample[:-1])
    Y.append("black")
for sample in x_white:
    X.append(sample[:-1])
    Y.append("white")
minMax = MinMaxScaler()
X = minMax.fit_transform(X)
x_train, x_test, y_train, y_label = train_test_split(X, Y, test_size=0.1, random_state=40)
num = len(x_train)
print(num)
for ra in ratio:
    print("DATA_RATIO:{}".format(ra))
    tem = int(num*ra)
    x_train = x_train[:tem]
    y_train = y_train[:tem]
    print('*********',len(x_train))
    print("RandomFroest")
    acc_all, pre_all, rec_all, fpr_all = 0, 0, 0, 0
    for i in range(3):
        y_test = model.RandomForest(x_train, y_train, x_test)
        acc, pre, rec, fpr = index.cal_index(y_test, y_label)
        acc_all += acc
        pre_all += pre
        rec_all += rec
        fpr_all += fpr
    print("All:\n", "Acc:", acc_all/3, "Pre:", pre_all/3, "Rec:", rec_all/3)
    acc_all, pre_all, rec_all, fpr_all = 0, 0, 0, 0
    print("DecisionTree")
    for i in range(3):
        y_test = model.DecsionTree(x_train, y_train, x_test)
        acc, pre, rec, fpr = index.cal_index(y_test, y_label)
        acc_all += acc
        pre_all += pre
        rec_all += rec
        fpr_all += fpr
    print("All:\n", "Acc:", acc_all/3, "Pre:", pre_all/3, "Rec:", rec_all/3)
    acc_all, pre_all, rec_all, fpr_all = 0, 0, 0, 0
    print("GradientBoosting")
    for i in range(3):
        y_test = model.GradientBoosting(x_train, y_train, x_test)
        acc, pre, rec, fpr = index.cal_index(y_test, y_label)
        acc_all += acc
        pre_all += pre
        rec_all += rec
        fpr_all += fpr
    print("All:\n", "Acc:", acc_all/3, "Pre:", pre_all/3, "Rec:", rec_all/3)
# f = open("result.txt", 'w')
# for i in range(len(x_test)):
#     str = x_name[i] + "," + y_test[i][0] + "\n"
#     f.write(str)
# f.close()
print("DS end")
