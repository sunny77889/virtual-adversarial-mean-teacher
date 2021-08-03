import csv
import os
import random
import pre
'''
特征提取
'''
print("data_product begin")
x_black = []
x_white = []
base_dir = "../../data_10_26/test/iscx process/dataProcess/target/output"
for path in os.listdir(base_dir):
    x_b = []
    x_w = []
    for i in os.listdir(base_dir + "/" + path):
        print(i)
        if i == '0':
            x_w = pre.pre_pcap(base_dir + '/' + path + '/' + i, path)
        if i == '1':
            x_b = pre.pre_pcap(base_dir + '/' + path + '/' + i, path)
    print(path, len(x_b), len(x_w))
    random.shuffle(x_w)
    x_w = x_w[:len(x_b)]
    x_black += x_b
    x_white += x_w
f = open("feature_list_b_3.csv", 'w+', newline='')
f_csv = csv.writer(f)
for key in x_black:
    f_csv.writerow(key)
f.close()
f = open("feature_list_w_3.csv", 'w+', newline='')
f_csv = csv.writer(f)
for key in x_white:
    f_csv.writerow(key)
f.close()
print("data_product end")




