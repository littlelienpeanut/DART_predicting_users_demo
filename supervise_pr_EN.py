import pandas as pd
import itertools
import csv
import random
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
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





def main():
    #variable
    print("Loading users' info")
    usernum = choose_user()
    data_v4 = load_data_v4(usernum)
    data_v5 = load_data_v5(usernum)
    data_all = load_data_all(data_v4, data_v5)
    pr_list = ['HH', 'Emo', 'Ext', 'Agr', 'Con', 'Ope']

    #big6
    print("Loading pr_info")
    HH, Emo, Ext, Agr, Con, Ope = load_pr()
    pr_label = {'HH':HH, 'Emo':Emo, 'Ext':Ext, 'Agr':Agr, 'Con':Con, 'Ope':Ope}

    #rmse score
    avg_rmse = {'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}


    #main
    user_data = data_all

    for c in range(1, 10001, 1):
        clf = linear_model.ElasticNet(random_state=2018, alpha= c*0.001)

        for pr in pr_list:
            loss = -cross_val_score(clf, user_data, pr_label[pr], cv=5, scoring='mean_squared_error')
            loss = sqrt_list(loss)
            rmse = np.mean(loss)
            avg_rmse[pr].append(rmse)

    print('data_all')
    for pr in pr_list:
        print(pr + ' /   Best testing score: ' + str('%.3f' % max(avg_rmse[pr])) + ' /  C : ' + str(avg_rmse[pr].index(max(avg_rmse[pr]))+1))


    #plot
    '''
    #HH, Emo, Ext, Agr, Con, Ope
    k = np.arange(129)

    plt.figure()
    plt.plot(k, rmse_pred["HH"], 'r-', label = 'testing')
    plt.plot(k, rmse_tr["HH"], 'b-', label = 'training')
    plt.plot(k, rmse_base["HH"], 'g-', label = 'baseline')
    plt.legend(loc = 'lower right')
    plt.ylabel("rmse")
    plt.xlabel("alpha value")
    plt.title("HH")
    plt.savefig("../HH.eps", format='eps', dpi=1000)

    plt.figure()
    plt.plot(k, rmse_pred["Emo"], 'r-', label = 'testing')
    plt.plot(k, rmse_tr["Emo"], 'b-', label = 'training')
    plt.plot(k, rmse_base["Emo"], 'g-', label = 'baseline')
    plt.legend(loc = 'lower right')
    plt.ylabel("rmse")
    plt.xlabel("alpha value")
    plt.title("Emo")
    plt.savefig("../Emo.eps", format='eps', dpi=1000)

    plt.figure()
    plt.plot(k, rmse_pred["Ext"], 'r-', label = 'testing')
    plt.plot(k, rmse_tr["Ext"], 'b-', label = 'training')
    plt.plot(k, rmse_base["Ext"], 'g-', label = 'baseline')
    plt.legend(loc = 'lower right')
    plt.ylabel("rmse")
    plt.xlabel("alpha value")
    plt.title("Ext")
    plt.savefig("../Ext.eps", format='eps', dpi=1000)

    plt.figure()
    plt.plot(k, rmse_pred["Agr"], 'r-', label = 'testing')
    plt.plot(k, rmse_tr["Agr"], 'b-', label = 'training')
    plt.plot(k, rmse_base["Agr"], 'g-', label = 'baseline')
    plt.legend(loc = 'lower right')
    plt.ylabel("rmse")
    plt.xlabel("alpha value")
    plt.title("Agr")
    plt.savefig("../Agr.eps", format='eps', dpi=1000)

    plt.figure()
    plt.plot(k, rmse_pred["Con"], 'r-', label = 'testing')
    plt.plot(k, rmse_tr["Con"], 'b-', label = 'training')
    plt.plot(k, rmse_base["Con"], 'g-', label = 'baseline')
    plt.legend(loc = 'lower right')
    plt.ylabel("rmse")
    plt.xlabel("alpha value")
    plt.title("Con")
    plt.savefig("../Con.eps", format='eps', dpi=1000)

    plt.figure()
    plt.plot(k, rmse_pred["Ope"], 'r-', label = 'testing')
    plt.plot(k, rmse_tr["Ope"], 'b-', label = 'training')
    plt.plot(k, rmse_base["Ope"], 'g-', label = 'baseline')
    plt.legend(loc = 'lower right')
    plt.ylabel("rmse")
    plt.xlabel("alpha value")
    plt.title("Ope")
    plt.savefig("../Ope.eps", format='eps', dpi=1000)

    plt.show()
    '''


if __name__ == '__main__':
    main()
