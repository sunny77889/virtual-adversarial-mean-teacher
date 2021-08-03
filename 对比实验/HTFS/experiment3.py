# -*- coding: UTF-8 -*-
from collections import Counter

import numpy as np
import os

from tensorflow_core.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Dense, Masking, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, \
    Reshape, Conv1D, ReLU
from tensorflow.keras.layers import LSTM
from tkinter import _flatten
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix

# from keras.utils import np_utils
import pandas as pd
import itertools

# gpus = tf.debugging.set_log_device_placement(True)
# np.random.seed(7)


malware_labels = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis', 'Shifu', 'Tinba', 'Virut', 'Zeus']
benign_labels = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'SMB', 'WorldOfWarcraft',
                 'Weibo']
'''
实验三：前述模型在少标记样本下的多分类效果对比。分别设置10类木马流量以及包含所有类的全集木马流量（ALL，将每类样本的准确率按照样本量加权平均获得），
标记样本占比为10%。
'''

def readdata():
    '''
    数据读取函数
    Returns:图片特征、包级特征、流级特征、标签

    '''
    datadir = "./feature_USTC-TFC2016/feature_27"
    flow_features = pd.DataFrame()
    img_features = []
    packet_features = []
    names = []
    i = 0
    labels = []
    for fpath in os.listdir(datadir):
        i = i + 1
        ftype = fpath.split('.')[1]
        name = fpath.split('_')[0]
        print(fpath)
        if name in malware_labels:
            names.append(name)
            if ftype == 'csv':
                f_csv = pd.read_csv(datadir + '/' + fpath, index_col=None)
                f_csv = f_csv.iloc[:, 1:]
                flow_features = flow_features.append(f_csv)
                labels.append([name for i in range(len(f_csv))])
            else:
                pp = fpath.split('.')[0]
                k = pp.split('_')[2]
                f_arr = np.load(datadir + '/' + fpath)
                f_list = f_arr.tolist()
                if k == 'img':
                    # np.concatenate(img_features, f_arr)
                    img_features = img_features + f_list
                elif k == 'packet':
                    # print('************', f_arr.shape)
                    packet_features = packet_features + f_list
    labels = np.array(list(_flatten(labels)))
    packet_features = np.array(packet_features)
    img_features = np.array(img_features)
    flow_features = np.array(flow_features)
    print(names)
    return img_features, packet_features, flow_features, labels

#
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def cnn_lstm(X1, X2, X3, y):
    '''
    模型构建与训练
    Args:
        X1: 图片特征
        X2: 包级特征
        X3: 流级特征
        y: 数据标签

    Returns:训练好得模型

    '''
    input1 = Input(shape=(28, 28, 1))
    a = Conv2D(32, (3, 3), strides=2, activation='relu')(input1)
    a = MaxPooling2D((2, 2))(a)
    a = Flatten()(a)
    a = Dense(24)(a)
    a = Reshape((2, 12))(a)
    a = Model(inputs=input1, outputs=a)

    input2 = Input(shape=(72,))
    b = Dense(72)(input2)
    b = Reshape((6, 12))(b)
    b = Model(inputs=input2, outputs=b)

    c = concatenate([a.output, b.output], axis=-2)
    c = LSTM(16)(c)

    input3 = Input(shape=(35,))
    d = Dense(32)(input3)
    d = Model(inputs=input3, outputs=d)

    e = concatenate([c, d.output], axis=-1)
    e = Dropout(0.3)(e)
    e = Dense(4, activation='softmax')(e)

    model = Model(inputs=[a.input, b.input, d.input], outputs=e)
    model.summary()
    apo = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=apo, metrics=['accuracy'])
    history = model.fit([X1, X2, X3], y, callbacks=[EarlyStopping(monitor='accuracy', patience=10, verbose=2, mode='max')],
              epochs=200, batch_size=128, verbose=2)

    epochs = range(len(history.history['accuracy']))
    plt.figure()
    plt.plot(epochs, history.history['accuracy'], 'b', label='Training acc')
    plt.plot(epochs, history.history['loss'], 'b', label='Training loss')
    plt.legend()
    plt.show()

    return model


# def experiment1():


if __name__ == '__main__':
    img_data, packet_data, flow_data, labels = readdata()
    print(img_data.shape)
    print(packet_data.shape)
    print(flow_data.shape)
    # 对图片特征进行归一化
    normalize_img_data = img_data / 255.0
    normalize_img_data = normalize_img_data.reshape(len(normalize_img_data), 28, 28, 1)

    # 对包级特征进行线性函数归一化
    data_2dim = packet_data.reshape(packet_data.shape[0] * packet_data.shape[1], packet_data.shape[2])
    normalizer = preprocessing.MinMaxScaler()
    tmp_data = normalizer.fit_transform(data_2dim)
    normalize_pack_data = tmp_data.reshape(packet_data.shape[0], packet_data.shape[1] * packet_data.shape[2])

    # 对流级特征进行归一化
    normalizer = preprocessing.MinMaxScaler()
    normalize_flow_data = normalizer.fit_transform(flow_data)

    labels = np.array(pd.get_dummies(labels))
    print(labels)

    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(normalize_img_data,
                                                                                                normalize_pack_data,
                                                                                                normalize_flow_data,
                                                                                                labels, test_size=0.1,
                                                                                                random_state=3)

    X1_train_s, _, X2_train_s, _, X3_train_s, _, y_train_s, _ = train_test_split(X1_train, X2_train, X3_train,
                                                                                 y_train, test_size=0.9,
                                                                                 random_state=4)

    htfs_model = cnn_lstm(X1_train_s, X2_train_s, X3_train_s, y_train_s)

    y_pre = htfs_model.predict([X1_test, X2_test, X3_test])
    y_pre = np.argmax(y_pre, axis=1)
    y_test = np.argmax(y_test, axis=1)


    names_pre = [malware_labels[y_pre[i]] for i in range(len(y_pre))]
    names_test = [malware_labels[y_test[i]] for i in range(len(y_test))]
    accuracy = accuracy_score(names_test, names_pre)
    print("Accuracy:%.2f%%" % (accuracy * 100))

    cnf_matrix = confusion_matrix(names_test, names_pre)

    n = len(cnf_matrix)

    for i in range(len(cnf_matrix[0])):
        rowsum, colsum = sum(cnf_matrix[i]), sum(cnf_matrix[r][i] for r in range(n))
        try:
            print(malware_labels[i],
                  'precision: %.2f%%' % ((cnf_matrix[i][i] / float(colsum))*100),
                  'recall: %.2f%%' % ((cnf_matrix[i][i] / float(rowsum))*100))
        except ZeroDivisionError:
            print(malware_labels[i],'precision: %.2f%%' % 0, 'recall: %.2f%%' % 0)

    plot_confusion_matrix(cnf_matrix, classes=malware_labels, normalize=True, title='Normalized confusion matrix')
