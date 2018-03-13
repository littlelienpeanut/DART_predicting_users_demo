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
    daily_v4 = []
    for fname in usernum:
        tmp_list = []
        data = pd.read_csv(fname + "_daily_v4.csv")
        for j in range(0, 24, 1):
            ### type("%.2f" % data['ratio'][j]) is string not float!!! ###
            tmp_list.append(data["ratio"][j])

        daily_v4.append(tmp_list)

    return daily_v4

def load_data_v5(usernum):
    daily_v5 = []
    cate_list = load_cate_list()

    for fname in usernum:
        tmp_list = []
        data = pd.read_csv(fname + "_daily_v5.csv")
        for cate in cate_list:
            tmp_list.append(float(data[cate][0]))
            
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

def sqrt_list(loss):
    tmp_list = []
    for i in range(len(loss)):
        tmp_list.append(sqrt(loss[i]))

    return tmp_list

def main():
    #variable
    print("Loading users' info")
    usernum = choose_user()
    data_v4 = load_data_v4(usernum)
    data_v5 = load_data_v5(usernum)
    rmse = []
    mse = []
    #big6
    print("Loading pr_info")
    HH, Emo, Ext, Agr, Con, Ope = load_pr()

    #main
    #knn
    '''
    for k in range(1, 30, 1):
        clf_knn = KNeighborsClassifier(n_neighbors = k)
    #clf_knn_pred = cross_val_predict(clf_knn, data_v5, HH, cv=5)
    ### cuz loss is a negative number, so must to add a negative sign ###
        loss = -cross_val_score(clf_knn, data_v4, HH, cv=5, scoring='mean_squared_error')
        loss = sqrt_list(loss)
        mse = np.mean(loss)
        rmse.append(mse)

    print(rmse)
    '''

    #LogisticRegression
    clf_log = LogisticRegression()
    clf_log_pred = cross_val_predict(clf_log, data_v4, HH, cv=5)
    print(clf_log_pred)

    '''
    loss = -cross_val_score(clf_log, data_v4, HH, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse = np.mean(loss)
    print(rmse)
    '''


if __name__ == '__main__':
    main()
