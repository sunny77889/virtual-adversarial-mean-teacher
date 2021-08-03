# -*- coding: UTF-8 -*-
import numpy as np
import os

from tensorflow_core.python.keras.callbacks import EarlyStopping

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
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

# from keras.utils import np_utils
import pandas as pd

# gpus = tf.debugging.set_log_device_placement(True)
# np.random.seed(7)


malware_labels = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis', 'Shifu', 'Tinba', 'Virut', 'Zeus']
benign_labels = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'SMB', 'WorldOfWarcraft',
                 'Weibo']

'''
实验二：检测未知样本的分类效果对比，即从十类样本中选取九类作为训练集，另一类作为测试集，总共重复十次
'''

def readdata():
    '''
    数据读取函数
    Returns:图片特征、包级特征、流级特征、标签

    '''
    test_name = 'Cridex'
    print(test_name)
    datadir = "./feature_USTC-TFC2016/feature_27"
    flow_features = pd.DataFrame()
    img_features = []
    packet_features = []
    flow_features_test = pd.DataFrame()
    img_features_test = []
    packet_features_test = []
    i = 0
    labels = []
    labels_test = []
    for fpath in os.listdir(datadir):
        i = i + 1
        ftype = fpath.split('.')[1]
        name = fpath.split('_')[0]
        print(name)
        print(fpath)
        if ftype == 'csv':
            f_csv = pd.read_csv(datadir + '/' + fpath, index_col=None)
            f_csv = f_csv.iloc[:, 1:]
            if name == test_name:
                flow_features_test = flow_features_test.append(f_csv)
                labels_test.append([1 for i in range(len(f_csv))])
            else:
                flow_features = flow_features.append(f_csv)
                if name in malware_labels:
                    labels.append([1 for i in range(len(f_csv))])
                else:
                    labels.append([0 for i in range(len(f_csv))])
        else:
            pp = fpath.split('.')[0]
            k = pp.split('_')[2]
            f_arr = np.load(datadir + '/' + fpath)
            f_list = f_arr.tolist()
            if k == 'img':
                if name == test_name:
                    img_features_test = img_features_test + f_list
                else:
                    img_features = img_features + f_list
            elif k == 'packet':
                if name == test_name:
                    packet_features_test = packet_features_test + f_list
                else:
                    packet_features = packet_features + f_list

    labels = np.array(list(_flatten(labels)))
    packet_features = np.array(packet_features)
    img_features = np.array(img_features)
    flow_features = np.array(flow_features)

    labels_test = np.array(list(_flatten(labels_test)))
    packet_features_test = np.array(packet_features_test)
    img_features_test = np.array(img_features_test)
    flow_features_test = np.array(flow_features_test)
    return img_features, packet_features, flow_features, labels, img_features_test, packet_features_test, flow_features_test, labels_test


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
    e = Dense(2, activation='softmax')(e)

    model = Model(inputs=[a.input, b.input, d.input], outputs=e)
    model.summary()
    apo = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=apo, metrics=['accuracy'])
    model.fit([X1, X2, X3], y, callbacks=[EarlyStopping(monitor='accuracy', patience=10, verbose=2, mode='max')],
              epochs=100, batch_size=128, verbose=2)
    return model


def preprocess(img_data, packet_data, flow_data):
    '''
    数据预处理：归一化
    Args:
        img_data: 图片特征
        packet_data:包级特征
        flow_data:流级特征

    Returns:归一化后得数据

    '''
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
    print(normalize_flow_data.shape)

    return normalize_img_data, normalize_pack_data, normalize_flow_data


if __name__ == '__main__':
    img_data, packet_data, flow_data, labels, X1_test, X2_test, X3_test, y_test = readdata()

    X1_train, X2_train, X3_train = preprocess(img_data, packet_data, flow_data)
    y_train = np.array(pd.get_dummies(labels))
    X1_test, X2_test, X3_test = preprocess(X1_test, X2_test, X3_test)

    htfs_model = cnn_lstm(X1_train, X2_train, X3_train, y_train)

    y_pre = htfs_model.predict([X1_test, X2_test, X3_test])
    y_pre = np.argmax(y_pre, axis=1)

    accuracy = accuracy_score(y_test, y_pre)
    recall = recall_score(y_test, y_pre)
    precision = precision_score(y_test, y_pre)
    print("Accuracy:%.2f%%" % (accuracy * 100))
    print('Precision:%.2f%%' % (precision * 100))
    print('Recall:%.2f%%' % (recall * 100))
