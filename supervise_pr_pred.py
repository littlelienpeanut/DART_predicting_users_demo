import pandas as pd
import itertools
import csv
import random
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def load_data_v4(usernum):
    daily_v4 = []

    for fname in usernum:
        tmp_list = []
        data = pd.read_csv(fname + "_daily_v4.csv")
        for j in range(0, 24, 1):
            tmp_list.append( "%.3f" % data["ratio"][j])
        daily_v4.append(tmp_list)

    return daily_v4

def load_data_v5(usernum):
    daily_v5 = []
    cate_list = load_cate_list()

    for fname in usernum:
        tmp_list = []
        data = pd.read_csv(fname + "_daily_v5.csv")
        for cate in cate_list:
            tmp_list.append( "%.3f" % data[cate][0])
        daily_v5.append(tmp_list)

    return daily_v5

def load_cate_list():
    list = []
    csv = pd.read_csv("cate_list_final.csv")
    for i in range(len(csv)):
        list.append(csv["cate"][i])
    return list

def load_user_list():
    user_list = pd.read_csv("user_list.csv")
    return user_list

def choose_user():
    usernum = []
    user_pr = pd.read_csv("user_pr.csv")
    user_id_list = load_user_list()
    for i in range(len(user_pr)):
        for j in range(len(user_id_list)):
            if user_pr["id"][i] == user_id_list["id"][j]:
                usernum.append("user" + str(j+1))
            else:
                pass

    return usernum

def load_pr():
    csv = pd.read_csv("user_pr.csv")
    #big6
    HH = []
    Emo = []
    Ext = []
    Agr = []
    Con = []
    Ope = []
    for i in range(len(csv)):
        HH.append(csv["HH"][i])
        Emo.append(csv["Emo"][i])
        Ext.append(csv["Ext"][i])
        Agr.append(csv["Agr"][i])
        Con.append(csv["Con"][i])
        Ope.append(csv["Ope"][i])

    return HH, Emo, Ext, Agr, Con, Ope


def main():
    #variable
    usernum = choose_user()
    data_v4 = load_data_v4(usernum)
    data_v5 = load_data_v5(usernum)
    #big6
    HH, Emo, Ext, Agr, Con, Ope = load_pr()

    #main
    clf_log = LogisticRegression(C = 2.0)
    clf_log_pred = cross_val_predict(clf_log, data_v4, HH, cv=5)


if __name__ == '__main__':
    main()
