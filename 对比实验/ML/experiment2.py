from data_read import data_read

import dataset
import model
import os
import random
import index
import time
'''
实验二：九类训练集，一类攻击作为测试集
return:
acc
'''
x_train = []
y_train = []
x_test = []
# 自测
y_test = []
y_label = []
X = []
Y = []

print("begin system")
print(time.asctime(time.localtime(time.time())))


x_black, x_white = data_read()
print(len(x_black), len(x_white))
print(len(x_black[0]))

black_set = ["Cridex-ALL", "Geodo-ALL", "Htbot-ALL", "Miuref-ALL", "Neris-ALL",
             "Nsis-ay-ALL", "Shifu-ALL", "Tinba-ALL", "Virut-ALL", "Zeus-ALL"]
for tem in range(10):
    ukclass = black_set[tem]
    num = 0
    print(ukclass)
    x_train = []
    x_test = []
    y_train = []
    y_label = []
    for key in x_black:
        if key[-1] == ukclass:
            num += 1
            x_test.append(key[:-1])
            y_label.append('black')
        else:
            x_train.append(key[:-1])
            y_train.append('black')
    print(ukclass, ':', len(x_test))
    for key in x_white:
        x_train.append(key[:-1])
        y_train.append('white')
    print("RandomFroest")
    for i in range(3):
        y_test = model.RandomForest(x_train, y_train, x_test)
        index.cal_acc(y_test, y_label)
    print("DecisionTree")
    for i in range(3):
        y_test = model.DecsionTree(x_train, y_train, x_test)
        index.cal_acc(y_test, y_label)
    print("GradientBoosting")
    for i in range(3):
        y_test = model.GradientBoosting(x_train, y_train, x_test)
        index.cal_acc(y_test, y_label)
print("DS end")
