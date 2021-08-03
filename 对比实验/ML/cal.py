import  numpy as np
from cipher_suite import cipher_suites
from cipher_suite import index
'''
计算各个值
'''

def cal(sequence):
    if not sequence:
        return 0, 0, 0, 0
    Max = max(sequence)
    Min = min(sequence)
    seq = np.array(sequence)
    mean = np.mean(seq)
    std = np.std(seq)
    return Max, Min, mean, std


def cal_seq(seq):
    tem = []
    for i, key in enumerate(seq):
        if i != 0:
            tem.append(key-seq[i-1])
    return tem


def cal_hex(seq):
    tem = []
    for key in seq:
        tem.append(hex(key))
    Sum = 0
    for key in tem:
        if key in cipher_suites:
            Sum += pow(2, cipher_suites[key])
    return Sum


def cal_ratio(seq):
    tem = 0
    total = 0
    for i, key in enumerate(seq):
        total += 4 * key
        tem += key * index[i]
    if tem != 0:
        tem = tem / total
    else:
        return 0
    return tem


def cal_fin(num):
    if num % 2 == 1:
        return True
    else:
        return False


def cal_syn(num):
    num = num // 2
    if num % 2 == 1:
        return True
    else:
        return False


def cal_rst(num):
    num = num // 4
    if num % 2 == 1:
        return True
    else:
        return False


def cal_psh(num):
    num = num // 8
    if num % 2 == 1:
        return True
    else:
        return False


def cal_ack(num):
    num = num // 16
    if num % 2 == 1:
        return True
    else:
        return False


def cal_urg(num):
    num = num // 32
    if num % 2 == 1:
        return True
    else:
        return False


def cal_ece(num):
    num = num // 64
    if num % 2 == 1:
        return True
    else:
        return False


def cal_cwe(num):
    num = num // 128
    if num % 2 == 1:
        return True
    else:
        return False


def cal_matrix(seq):
    a = np.zeros((15, 15), dtype=int)
    for i, key in enumerate(seq):
        if i < len(seq)-1:
            tem1 = key // 150
            tem2 = seq[i + 1] // 150
            tem1 = tem1 if tem1 < 15 else 14
            tem2 = tem2 if tem2 < 15 else 14
            a[tem1][tem2] += 1
    sum = np.sum(a)
    a = a/sum
    return a


def cal_div(num1, num2):
    if num2 != 0:
        return num1/num2
    else:
        return 0