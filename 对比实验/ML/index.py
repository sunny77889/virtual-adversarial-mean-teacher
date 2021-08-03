from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
'''
计算实验指标
'''
def cal_index(y_test, y_label):
    acc = 0
    n = len(y_test)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(n):
        if y_label[i] == "black":
            if y_test[i] == "black":
                TN += 1
            if y_test[i] == "white":
                FP += 1
        else:
            if y_test[i] == "black":
                FN += 1
            if y_test[i] == "white":
                TP += 1
        if y_test[i] == y_label[i]:
            acc += 1
    check_out = TP/(FN+TP)
    false_positive = FP/(FP+TN)
    grade = check_out - false_positive
    grade *= 100
    Acc = (TN + TP) / (TN + FP + TP + FN)
    Pre = TP / (FP + TP)
    Rec = TP / (TP + FN)
    Fpr = FP / (TN + FP)
    # print("TP:", TP, "FN:", FN, "FP:", FP, "TN:", TN)
    print("Acc:%.4f" % Acc, "Pre:%.4f" % Pre, "Rec:%.4f" % Rec, "Fpr:%.4f" % Fpr)
    return Acc, Pre, Rec, Fpr
    # print(accuracy_score(y_label, y_test))
    # print(precision_score(y_label, y_test))
    # print(recall_score(y_label, y_test))



def cal_index_1(y_test, y_label):
    acc = 0
    n = len(y_test)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(n):
        if y_label[i] == 1:
            if y_test[i] == 1:
                TP += 1
            if y_test[i] == 0:
                FN += 1
        else:
            if y_test[i] == 1:
                FP += 1
            if y_test[i] == 0:
                TN += 1
        if y_test[i] == y_label[i]:
            acc += 1
    check_out = TP/(FN+TP)
    false_positive = FP/(FP+TN)
    grade = check_out - false_positive
    grade *= 100

    print("grade:", grade)
    acc = acc / n
    print("acc:", acc)

    print("index end")


def cal_acc(y_test, y_label):
    n = len(y_test)
    acc = 0
    for i in range(n):
        if y_label[i] == y_test[i]:
            acc += 1
    print("acc:{}".format(acc/n))


def cal_acc_type(num_set, y_test, y_label):
    # score_set = {"Cridex-ALL": 0, "Geodo-ALL": 0, "Htbot-ALL": 0, "Miuref-ALL": 0, "Neris-ALL": 0,
    #              "Nsis-ay-ALL": 0, "Shifu-ALL": 0, "Tinba-ALL": 0, "Virut-ALL": 0, "Zeus-ALL": 0, "All": 0}
    score_set = {"13": 0, "14": 0, "15": 0, "17": 0, "All": 0}
    for i in range(len(y_test)):
        if y_test[i] == y_label[i]:
            score_set[y_test[i]] += 1
            score_set['All'] += 1
    for key, value in num_set.items():
        score_set[key] /= value
    print(score_set)
