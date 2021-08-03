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
实验一：在少标记样本下（10%、5%、1%、0.1%标记样本）的二分类（木马流量，正常流量）效果对比
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
    i = 0
    labels = []
    malware_list1 = []
    malware_list2 = []
    benign_list1 = []
    benign_list2 = []
    for fpath in os.listdir(datadir):
        i = i + 1
        ftype = fpath.split('.')[1]
        name = fpath.split('-')[0]
        print(fpath)
        if ftype == 'csv':
            f_csv = pd.read_csv(datadir + '/' + fpath, index_col=None)
            f_csv = f_csv.iloc[:, 1:]
            print(np.array(f_csv).shape)
            flow_features = flow_features.append(f_csv)
            if name in malware_labels:
                malware_list1.append(name)
                malware_list2.append(len(f_csv))
                print(name, len(f_csv))
                labels.append([1 for i in range(len(f_csv))])
            else:
                benign_list1.append(name)
                benign_list2.append(len(f_csv))
                print(name, len(f_csv))
                labels.append([0 for i in range(len(f_csv))])
        else:
            pp = fpath.split('.')[0]
            k = pp.split('_')[1]
            f_arr = np.load(datadir + '/' + fpath)
            f_list = f_arr.tolist()
            if k == 'img':
                img_features = img_features + f_list
            elif k == 'packet':
                packet_features = packet_features + f_list
    print('malware_name', malware_list1, 'malware_num', malware_list2, 'benign_name', benign_list1, 'benign_num',
          benign_list2)
    name_df = pd.DataFrame({'malware_name': malware_list1, 'malware_num': malware_list2, 'benign_name': benign_list1,
                            'benign_num': benign_list2})
    name_df.to_csv('name_num.csv')
    labels = np.array(list(_flatten(labels)))
    packet_features = np.array(packet_features)
    img_features = np.array(img_features)
    flow_features = np.array(flow_features)
    return img_features, packet_features, flow_features, labels

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
    model.fit([X1, X2, X3], y, epochs=200, batch_size=128, verbose=2) #训练
    return model


if __name__ == '__main__':
    label_percent = 0.1  # 标注比例
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
    print(normalize_flow_data.shape)

    labels = np.array(pd.get_dummies(labels))

    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(normalize_img_data,
                                                                                                normalize_pack_data,
                                                                                                normalize_flow_data,
                                                                                                labels, test_size=0.1,
                                                                                                random_state=3)

    X1_train_s, _, X2_train_s, _, X3_train_s, _, y_train_s, _ = train_test_split(X1_train, X2_train, X3_train,
                                                                                 y_train, test_size=1 - label_percent,
                                                                                 random_state=4)

    htfs_model = cnn_lstm(X1_train_s, X2_train_s, X3_train_s, y_train_s)

    # 评估测试集
    y_pre = htfs_model.predict([X1_test, X2_test, X3_test])
    y_pre = np.argmax(y_pre, axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pre)
    recall = recall_score(y_test, y_pre)
    precision = precision_score(y_test, y_pre)
    f1 = f1_score(y_test, y_pre)
    print("Accuracy:%.2f%%" % (accuracy * 100))
    print('Precision:%.2f%%' % (precision * 100))
    print('Recall:%.2f%%' % (recall * 100))
    print('f1:%.2f%%' % (f1 * 100))
