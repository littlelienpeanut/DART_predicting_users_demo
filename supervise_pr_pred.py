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

def load_data_all(v4, v5):
    tmp_v4 = v4
    tmp_v5 = v5
    data_all = []
    for kmm_i in range(len(v4)):
        tmp_list = []
        tmp_list = tmp_v4[kmm_i] + tmp_v5[kmm_i]
        data_all.append(tmp_list)

    return data_all

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

def rmse_mean(pr):
    rmse = []

    split_ind = [0, 103, 206, 309, 412, 513]
    for i in range(len(split_ind)-1):
        test = []
        train = []
        for j in range(len(pr)):
            if j >= split_ind[i] and j <= split_ind[i+1] - 1:
                test.append(pr[j])
            else:
                train.append(pr[j])

        test = np.asarray(test)
        mean = np.mean(train)
        mean = float('%.2f' % (mean))
        mse = np.mean(np.square(test - mean))
        rmse.append(np.sqrt(mse))

    rmse = np.asarray(rmse)
    output = np.mean(rmse)

    return output


def main():
    #variable
    print("Loading users' info")
    usernum = choose_user()
    data_v4 = load_data_v4(usernum)
    data_v5 = load_data_v5(usernum)
    data_all = load_data_all(data_v4, data_v5)
    pr_list = ['HH', 'Emo', 'Ext', 'Agr', 'Con', 'Ope']
    rmse_pred = {'HH':0, 'Emo':0, 'Ext':0, 'Agr':0, 'Con':0, 'Ope':0}
    rmse_base = {'HH':0, 'Emo':0, 'Ext':0, 'Agr':0, 'Con':0, 'Ope':0}
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
    '''
    clf_log = LogisticRegression()
    clf_log_pred = cross_val_predict(clf_log, data_v4, HH, cv=5)
    print(clf_log_pred)
    '''

    #HH, Emo, Ext, Agr, Con, Ope
    '''
    clf_log = linear_model.LinearRegression(n_jobs=-1)
    loss = -cross_val_score(clf_log, data_v4, HH, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['HH'] = "%.3f" % np.mean(loss)
    rmse_base['HH'] = "%.3f" % rmse_mean(HH)

    loss = -cross_val_score(clf_log, data_v4, Emo, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Emo'] = "%.3f" % np.mean(loss)
    rmse_base['Emo'] = "%.3f" % rmse_mean(Emo)

    loss = -cross_val_score(clf_log, data_v4, Ext, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Ext'] = "%.3f" % np.mean(loss)
    rmse_base['Ext'] = "%.3f" % rmse_mean(Ext)

    loss = -cross_val_score(clf_log, data_v4, Agr, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Agr'] = "%.3f" % np.mean(loss)
    rmse_base['Agr'] = "%.3f" % rmse_mean(Agr)

    loss = -cross_val_score(clf_log, data_v4, Con, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Con'] = "%.3f" % np.mean(loss)
    rmse_base['Con'] = "%.3f" % rmse_mean(Con)

    loss = -cross_val_score(clf_log, data_v4, Ope, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Ope'] = "%.3f" % np.mean(loss)
    rmse_base['Ope'] = "%.3f" % rmse_mean(Ope)

    print("rmse of data_v4 in pr_list")
    for pr in pr_list:
        print(rmse_pred[pr])

    print("")
    print("rmse of mean of pr")
    for pr in pr_list:
        print(rmse_base[pr])
    '''

    #HH, Emo, Ext, Agr, Con, Ope
    clf_log = linear_model.LinearRegression()
    loss = -cross_val_score(clf_log, data_v5, HH, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['HH'] = "%.3f" % np.mean(loss)
    rmse_base['HH'] = "%.3f" % rmse_mean(HH)

    loss = -cross_val_score(clf_log, data_v5, Emo, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Emo'] = "%.3f" % np.mean(loss)
    rmse_base['Emo'] = "%.3f" % rmse_mean(Emo)

    loss = -cross_val_score(clf_log, data_v5, Ext, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Ext'] = "%.3f" % np.mean(loss)
    rmse_base['Ext'] = "%.3f" % rmse_mean(Ext)

    loss = -cross_val_score(clf_log, data_v5, Agr, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Agr'] = "%.3f" % np.mean(loss)
    rmse_base['Agr'] = "%.3f" % rmse_mean(Agr)

    loss = -cross_val_score(clf_log, data_v5, Con, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Con'] = "%.3f" % np.mean(loss)
    rmse_base['Con'] = "%.3f" % rmse_mean(Con)

    loss = -cross_val_score(clf_log, data_v5, Ope, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Ope'] = "%.3f" % np.mean(loss)
    rmse_base['Ope'] = "%.3f" % rmse_mean(Ope)

    print("rmse of data_v5 in pr_list")
    for pr in pr_list:
        print(rmse_pred[pr])

    print("")
    print("rmse of mean of pr")
    for pr in pr_list:
        print(rmse_base[pr])

    #HH, Emo, Ext, Agr, Con, Ope
    '''
    clf_log = linear_model.LinearRegression()
    loss = -cross_val_score(clf_log, data_all, HH, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['HH'] = "%.3f" % np.mean(loss)
    rmse_base['HH'] = "%.3f" % rmse_mean(HH)

    loss = -cross_val_score(clf_log, data_all, Emo, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Emo'] = "%.3f" % np.mean(loss)
    rmse_base['Emo'] = "%.3f" % rmse_mean(Emo)

    loss = -cross_val_score(clf_log, data_all, Ext, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Ext'] = "%.3f" % np.mean(loss)
    rmse_base['Ext'] = "%.3f" % rmse_mean(Ext)

    loss = -cross_val_score(clf_log, data_all, Agr, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Agr'] = "%.3f" % np.mean(loss)
    rmse_base['Agr'] = "%.3f" % rmse_mean(Agr)

    loss = -cross_val_score(clf_log, data_all, Con, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Con'] = "%.3f" % np.mean(loss)
    rmse_base['Con'] = "%.3f" % rmse_mean(Con)

    loss = -cross_val_score(clf_log, data_all, Ope, cv=5, scoring='mean_squared_error')
    loss = sqrt_list(loss)
    rmse_pred['Ope'] = "%.3f" % np.mean(loss)
    rmse_base['Ope'] = "%.3f" % rmse_mean(Ope)

    print("rmse of data_all in pr_list")
    for pr in pr_list:
        print(rmse_pred[pr])

    print("")
    print("rmse of mean of pr")
    for pr in pr_list:
        print(rmse_base[pr])
    '''


if __name__ == '__main__':
    main()
