import pre
import _json
import json
import numpy as np
import csv
print("data_product begin")
# x_black = pre.pre_pcap("data/Cridex-All/black/")

# x_black = pre.pre_pcap("../../data_10_11/file/PcapToMnist/2_Session/AllLayers/black/")
x_black = pre.pre_pcap("../../data_10_26/iscx process/dataProcess/target/output/testbed-13jun/1")
print("half")
# x_white = pre.pre_pcap("data/Cridex-All/white/")
# x_white =pre.pre_pcap("../../data_10_11/file/PcapToMnist/2_Session/AllLayers/white/")
x_white = pre.pre_pcap("../../data_10_26/iscx process/dataProcess/target/output/testbed-13jun/0")
f = open("feature_list_b_2.csv", 'w+', newline='')
f_csv = csv.writer(f)
for key in x_black:
    f_csv.writerow(key)
f.close()
f = open("feature_list_w_2.csv", 'w+', newline='')
f_csv = csv.writer(f)
for key in x_white:
    f_csv.writerow(key)
f.close()
print("data_product end")




