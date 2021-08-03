import pre
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_read import data_read

x_black, x_white = data_read()
x_train = []
y_train = []
x_test = []
y_label = []
black_set = ["Cridex-All", "Geodo-All", "Htbot-All", "Miuref-ALL", "Neris-ALL",
             "Nsis-ay-ALL", "Shifu-ALL", "Tinba-ALL", "Virut-ALL", "Zeus-ALL"]
white_set = ["BitTorrent-ALL", "Facetime-ALL", "FTP-ALL", "Gmail-ALL", "MySQL-ALL",
             "Outlook-ALL", "Skype-ALL", "SMB-ALL", "Weibo-ALL", "WorldOfWarcraft-ALL"]
shiyan1 = True


def pre_data():

    X = []
    Y = []
    minMax = MinMaxScaler()
    for sample in x_black:
        # X.append(sample)
        # Y.append('black')
        X.append(sample[:-1])
        if shiyan1:
            if sample[-1] in black_set:
                Y.append("black")
            else:
                Y.append("white")
        else:
            Y.append(sample[-1])
    for sample in x_white:
        # X.append(sample)
        # Y.append('white')
        X.append(sample[:-1])
        if shiyan1:
            if sample[-1] in black_set:
                Y.append("black")
            else:
                Y.append("white")
        else:
            Y.append(sample[-1]) 
    X = minMax.fit_transform(X)
    x_train, x_test, y_train, y_label = train_test_split(X, Y, test_size=0.1, random_state=40)
    num = 40000
    ratio = 0.01
    print("DATA_RATIO:{}".format(ratio))
    tem = int(num*ratio)
    x_train = x_train[:tem]
    y_train = y_train[:tem]
    # x_train, x_test, y_train, y_label = train_test_split(X, Y, test_size=0.33, random_state=42)
    print("pre_data end")
    return(x_train, y_train, x_test, y_label)


if __name__ =="__main__":
    pre_data()
