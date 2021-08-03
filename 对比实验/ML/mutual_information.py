import dataset
import numpy as np
from sklearn import metrics
if __name__ == '__main__':
    feature_list = ["pack_num", "time", "dport", "flow_num", "ip_src", "cipher_num", "packetsize_size",
                "max_time", "min_time", "mean_time", "std_time", "max_time_src", "min_time_src", "mean_time_src", "std_time_src",
                "max_time_dst", "min_time_dst", "mean_time_dst", "std_time_dst", "max_time_flow", "min_time_flow", "mean_time_flow", "std_time_flow",
                "max_packetsize_packet", "min_packetsize_packet", "mean_packetsize_packet", "std_packetsize_packet",
                "max_packetsize_src", "min_packetsize_src", "mean_packetsize_src", "std_packetsize_src",
                "max_packetsize_dst", "min_packetsize_dst", "mean_packetsize_dst", "std_packetsize_dst",
                "max_packetsize_flow", "min_packetsize_flow", "mean_packetsize_flow", "std_packetsize_flow",
                "cipher", "cipher_num", "cipher_content_ratio"]
    x_train, y_train, x_test, y_label = dataset.pre_data()
    x = x_train + x_test
    y = y_train + y_label
    tem = np.array(x)

    for i in range(len(tem[0])):
        print(tem[:, i])
        mutual = metrics.mutual_info_score(tem[:, i], y)
        print(feature_list[i], mutual)
