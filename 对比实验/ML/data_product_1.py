import pre
import _json
import json
import numpy as np
import csv
print("data_product begin")
pcap_dir='pcap_data'
out_dir='data_features'
for filename in os.listdir(pcap_dir):
    x_black = pre.pre_pcap(pcap_dir+'/'+filename, file_name)
    f = open(out_dir+'/'+filename+".csv", 'w+', newline='')
    f_csv = csv.writer(f)
    for key in x_black:
        f_csv.writerow(key)
    f.close()
print("data_product end")
