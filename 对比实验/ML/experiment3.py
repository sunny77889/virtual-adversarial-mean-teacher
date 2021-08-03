from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
# import tensorflow as tf
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_read import data_read
x_black, x_white = data_read()
import pre
import dataset
import model
import os
import random
import index
import time
'''
实验3 多分类实验
return:
acc
'''
x_train = []
y_train = []
x_test = []
# 自测
y_test = []
y_label = []
# X = []
# Y = []

print("begin system")
print(time.asctime(time.localtime(time.time())))

black_set = ["Cridex-ALL", "Geodo-ALL", "Htbot-ALL", "Miuref-ALL", "Neris-ALL",
             "Nsis-ay-ALL", "Shifu-ALL", "Tinba-ALL", "Virut-ALL", "Zeus-ALL"]
num_dic = {"Cridex-ALL": 0, "Geodo-ALL": 0, "Htbot-ALL": 0, "Miuref-ALL": 0, "Neris-ALL": 0,
             "Nsis-ay-ALL": 0, "Shifu-ALL": 0, "Tinba-ALL": 0, "Virut-ALL": 0, "Zeus-ALL": 0, "All": 0}
# lightGBM
# y_test = model.LightGBM(x_train, y_train, x_test, y_label)
# index.cal_index_1(y_test, y_label)
X = []
Y = []
minMax = MinMaxScaler()
for sample in x_black:
    X.append(sample[:-1])
    Y.append(sample[-1])
X = minMax.fit_transform(X)
x_train, x_test, y_train, y_label = train_test_split(X, Y, test_size=0.1, random_state=40)
for sample in y_label:
    num_dic[sample] += 1
    num_dic['All'] += 1

num = 40000
ratio = 0.1
print("DATA_RATIO:{}".format(ratio))
tem = int(num*ratio)
x_train = x_train[:tem]
y_train = y_train[:tem]
print("RandomFroest")
for i in range(3):
    y_test = model.RandomForest(x_train, y_train, x_test)
    index.cal_acc_type(num_dic, y_test, y_label)
print("DecisionTree")
for i in range(3):
    y_test = model.DecsionTree(x_train, y_train, x_test)
    index.cal_acc_type(num_dic, y_test, y_label)
print("GradientBoosting")
for i in range(3):
    y_test = model.GradientBoosting(x_train, y_train, x_test)
    index.cal_acc_type(num_dic, y_test, y_label)
# f = open("result.txt", 'w')
# for i in range(len(x_test)):
#     str = x_name[i] + "," + y_test[i][0] + "\n"
#     f.write(str)
# f.close()
print("DS end")
