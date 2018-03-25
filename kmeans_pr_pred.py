import pandas as pd
import itertools
import csv
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
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
import matplotlib.pyplot as plt
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
    split_ind = np.linspace(0, len(pr), num=6, dtype=np.int)
    split_ind = split_ind.tolist()
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

def cv_training_rmse(clf, data, label):
    rmse = []
    split_ind = np.linspace(0, len(data), num=6, dtype=np.int)
    split_ind = split_ind.tolist()
    for i in range(len(split_ind)-1):
        tr_x = []
        tr_y = []
        te_x = []
        te_y = []
        for j in range(len(data)):
            if j >= split_ind[i] and j <= split_ind[i+1] - 1:
                te_x.append(data[j])
                te_y.append(label[j])
            else:
                tr_x.append(data[j])
                tr_y.append(label[j])

        tr_clf = clf.fit(tr_x, tr_y)
        tr_pred = tr_clf.predict(tr_x)
        for tr_i in range(len(tr_y)):
            tr_y[tr_i] = tr_y[tr_i] - tr_pred[tr_i]
        tr_y = np.asarray(tr_y)
        mse = np.mean(np.square(tr_y))
        rmse.append(np.sqrt(mse))

    rmse = np.asarray(rmse)
    output = np.mean(rmse)

    return output

def cv_testing_rmse(clf, data, label):
    rmse = []
    split_ind = np.linspace(0, len(data), num=6, dtype=np.int)
    split_ind = split_ind.tolist()
    for i in range(len(split_ind)-1):
        tr_x = []
        tr_y = []
        te_x = []
        te_y = []
        for j in range(len(data)):
            if j >= split_ind[i] and j <= split_ind[i+1] - 1:
                te_x.append(data[j])
                te_y.append(label[j])
            else:
                tr_x.append(data[j])
                tr_y.append(label[j])

        tr_clf = clf.fit(tr_x, tr_y)
        te_pred = tr_clf.predict(te_x)
        #print(te_pred)
        for tr_i in range(len(te_y)):
            te_y[tr_i] = te_y[tr_i] - te_pred[tr_i]
        te_y = np.asarray(te_y)
        mse = np.mean(np.square(te_y))
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
    kms_data = []
    data_all = load_data_all(data_v4, data_v5)
    pr = {'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}
    pr_list = ['HH', 'Emo', 'Ext', 'Agr', 'Con', 'Ope']
    rmse_pred = {'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}
    rmse_base = {'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}
    rmse_tr = {'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}
    tmp_rmse_pred = {}
    tmp_rmse_base = {}
    tmp_rmse_tr = {}

    mse = []

    #big6
    print("Loading pr_info")
    HH, Emo, Ext, Agr, Con, Ope = load_pr()
    for user_num in range(len(HH)):
        pr['HH'].append(HH[user_num])
        pr['Emo'].append(Emo[user_num])
        pr['Ext'].append(Ext[user_num])
        pr['Agr'].append(Agr[user_num])
        pr['Con'].append(Con[user_num])
        pr['Ope'].append(Ope[user_num])



    #main
    #hyper parameter settings
    #how many clusters?
    n_clusters = 5

    #which dataset?
    user_data = data_v4


    # ------------------------------------------------------------------------#
    #kms model training
    for km4_i in range(len(user_data)):
        kms_data.append(user_data[km4_i])

    kms_model = KMeans(n_clusters, random_state=42).fit(kms_data)

    ### If you want to check detail userid_daily_v4.csv, you have to find the user_id in pr then go to user_list to get the user"id"_daily_v4.csv. ###
    kms_data = {}
    kms_pr = {}
    kms_user_list = {}

    for c_num in range(n_clusters):
        kms_data.update({str(c_num):[]})
        kms_user_list.update({str(c_num):[]})
        tmp_rmse_pred.update({str(c_num):{'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}})
        tmp_rmse_base.update({str(c_num):{'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}})
        tmp_rmse_tr.update({str(c_num):{'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}})

    for c_num in range(n_clusters):
        kms_pr.update({str(c_num):{'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}})

    for o_i in range(len(user_data)):
        kms_data[str(kms_model.labels_[o_i])].append(user_data[o_i])
        kms_user_list[str(kms_model.labels_[o_i])].append(o_i)
        for pr_list_i in pr_list:
            kms_pr[str(kms_model.labels_[o_i])][pr_list_i].append(pr[pr_list_i][o_i])

    for c_num in range(n_clusters):
        for alpha_val in range(1, 13, 1):
            clf = linear_model.Ridge(alpha = alpha_val)

            for pr_list_i in pr_list:
                tmp_rmse_pred[str(c_num)][pr_list_i].append(cv_testing_rmse(clf, kms_data[str(c_num)], kms_pr[str(c_num)][pr_list_i]))
                tmp_rmse_base[str(c_num)][pr_list_i].append(rmse_mean(kms_pr[str(c_num)][pr_list_i]))
                tmp_rmse_tr[str(c_num)][pr_list_i].append(cv_training_rmse(clf, kms_data[str(c_num)], kms_pr[str(c_num)][pr_list_i]))

        for pr_list_i in pr_list:
            if c_num == 0:
                rmse_pred[pr_list_i] = tmp_rmse_pred[str(c_num)][pr_list_i]
                rmse_base[pr_list_i] = tmp_rmse_base[str(c_num)][pr_list_i]
                rmse_tr[pr_list_i] = tmp_rmse_tr[str(c_num)][pr_list_i]
            else:
                for ind_i in range(len(tmp_rmse_pred[str(c_num)][pr_list_i])):
                    rmse_pred[pr_list_i][ind_i] += tmp_rmse_pred[str(c_num)][pr_list_i][ind_i]
                    rmse_base[pr_list_i][ind_i] += tmp_rmse_base[str(c_num)][pr_list_i][ind_i]
                    rmse_tr[pr_list_i][ind_i] += tmp_rmse_tr[str(c_num)][pr_list_i][ind_i]

    for pr_i in pr_list:
        rmse_pred[pr_i][:] = [float('%.3f' % (float(x) / float(n_clusters))) for x in rmse_pred[pr_i]]
        rmse_base[pr_i][:] = [float('%.3f' % (float(x) / float(n_clusters))) for x in rmse_base[pr_i]]
        rmse_tr[pr_i][:] = [float('%.3f' % (float(x) / float(n_clusters))) for x in rmse_tr[pr_i]]

    print("rmse of data_all in testing")
    for pr in pr_list:
        print(rmse_pred[pr])

    print("")
    print("rmse of data_all training")
    for pr in pr_list:
        print(rmse_tr[pr])

    print("")
    print("rmse of data_all baseline")
    for pr in pr_list:
        print(rmse_base[pr])


    #plot
    #HH, Emo, Ext, Agr, Con, Ope
    k = np.arange(12)

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




if __name__ == '__main__':
    main()
