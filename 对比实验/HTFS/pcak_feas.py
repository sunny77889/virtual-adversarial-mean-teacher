from scapy.all import *
import numpy as np
import os
FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20
ECE = 0x40
CWR = 0x80

def packet_show(pcaps,i):
    packet = pcaps[i].payload
    features=[0]*12
    if packet.haslayer(IP):
        if packet[IP].flags=='MF':
            features[0]=1
        if packet[IP].flags=='DF':
            features[1]=1
        features[2]=packet[IP].ttl
    if packet.haslayer(TCP):
        if packet[TCP].flags&FIN:
            features[3]=1
        if packet[TCP].flags&SYN:
            features[4]=1
        if packet[TCP].flags&RST:
            features[5]=1
        if packet[TCP].flags&PSH:
            features[6]=1
        if packet[TCP].flags&ACK:
            features[7]=1
        if packet[TCP].flags&URG:
            features[8]=1
        if packet[TCP].flags&ECE:
            features[9]=1
        if packet[TCP].flags&CWR:
            features[10]=1
    return features
all_feas=[]
for file in os.listdir('./pcaps/'):
    kk='./pcaps/'+file
    pcaps = rdpcap(kk)
    feas=[]
    for i in range(len(pcaps)):
        if i<6:
            feas.append(packet_show(pcaps, i))
            
    if len(pcaps)<6:
        for j in range(6-len(pcaps)):
            feas.append([0]*12)
    all_feas.append(feas)        
all_feas=np.array(all_feas)
np.save('pack_feas.npy', all_feas)
print(all_feas.shape)