import json
from sklearn.preprocessing import MinMaxScaler
import csv
'''
读入文件
'''
def data_read():
    # minMax = MinMaxScaler()
    with open('feature_list_b.csv', 'r') as f:
        # x_black = json.loads(tem)
        reader = csv.reader(f)
        x_black = list(reader)
        # x_black = minMax.fit_transform(x_black)
    f.close()
    with open('feature_list_w.csv', 'r') as f:
        # x_white = json.loads(tem)
        reader = csv.reader(f)
        x_white = list(reader)
        # x_white = minMax.fit_transform(x_white)
    f.close()
    return x_black, x_white