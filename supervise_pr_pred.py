import pandas as pd
import itertools
import csv
import random
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.learning_curve import learning_curve
from sklearn.metrics import mean_squared_error


def load_data_v4(usernum):
    #can not just append float to list, must have to change the list to array and set dtype = float64.
    daily_v4 = []
    for fname in usernum:
        tmp_list = []
        tmp_list = np.array(tmp_list)
        data = pd.read_csv(fname + "_daily_v4.csv")
        for j in range(0, 24, 1):
            tmp_list = np.append(tmp_list, "%.3f" % data["ratio"][j])

        tmp_list.dtype = 'float'
        daily_v4.append(tmp_list)

    return daily_v4

def load_data_v5(usernum):
    daily_v5 = []
    cate_list = load_cate_list()

    for fname in usernum:
        tmp_list = []
        tmp_list = np.asarray(tmp_list)
        data = pd.read_csv(fname + "_daily_v5.csv")
        for cate in cate_list:
            tmp_list = np.append(tmp_list, "%.3f" % float(data[cate][0]))

        tmp_list.dtype = 'float64'
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

    HH = np.array(HH)
    Emo = np.array(Emo)
    Ext = np.array(Ext)
    Agr = np.array(Agr)
    Con = np.array(Con)
    Ope = np.array(Ope)

    for i in range(len(csv)):
        HH = np.append(HH, csv["HH"][i])
        Emo = np.append(Emo, csv["Emo"][i])
        Ext = np.append(Ext, csv["Ext"][i])
        Agr = np.append(Agr, csv["Agr"][i])
        Con = np.append(Con, csv["Con"][i])
        Ope = np.append(Ope, csv["Ope"][i])

    return HH, Emo, Ext, Agr, Con, Ope


def main():
    #variable
    print("Loading users' info")
    usernum = choose_user()
    data_v4 = load_data_v4(usernum)
    data_v5 = load_data_v5(usernum)
    rmse = []
    #big6
    print("Loading pr_info")
    HH, Emo, Ext, Agr, Con, Ope = load_pr()

    #main
    for k in range(1, 20, 1):
        clf_knn = KNeighborsClassifier(n_neighbors = k)
        #cuz loss is a negative number, so must to add a minus
        loss = -cross_val_score(clf_knn, data_v4, HH, cv=5, scoring='mean_squared_error')
        mse = loss.mean()
        rmse.append(sqrt(mse))

    print(rmse)

if __name__ == '__main__':
    main()
