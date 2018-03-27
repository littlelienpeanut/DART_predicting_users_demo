import pandas as pd
import itertools
import csv
from sklearn.linear_model import Ridge
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
        # print(tr_pred)
        # print("")
        # print(tr_y)
        # input()
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
    data_all = load_data_all(data_v4, data_v5)
    pr_list = ['HH', 'Emo', 'Ext', 'Agr', 'Con', 'Ope']
    rmse_pred = {'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}
    rmse_base = {'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}
    rmse_tr = {'HH':[], 'Emo':[], 'Ext':[], 'Agr':[], 'Con':[], 'Ope':[]}

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
    clf = LogisticRegression()
    clf_pred = cross_val_predict(clf, data_v4, HH, cv=5)
    print(clf_pred)
    '''

    #HH, Emo, Ext, Agr, Con, Ope
    '''
    clf = svm.SVC()
    rmse_pred['HH'] = "%.3f" % cv_testing_rmse(clf, data_v4, HH)
    rmse_base['HH'] = "%.3f" % rmse_mean(HH)
    rmse_tr['HH'] = "%.3f" % cv_training_rmse(clf, data_v4, HH)

    rmse_pred['Emo'] = "%.3f" % cv_testing_rmse(clf, data_v4, Emo)
    rmse_base['Emo'] = "%.3f" % rmse_mean(Emo)
    rmse_tr['Emo'] = "%.3f" % cv_training_rmse(clf, data_v4, Emo)

    rmse_pred['Ext'] = "%.3f" % cv_testing_rmse(clf, data_v4, Ext)
    rmse_base['Ext'] = "%.3f" % rmse_mean(Ext)
    rmse_tr['Ext'] = "%.3f" % cv_training_rmse(clf, data_v4, Ext)

    rmse_pred['Agr'] = "%.3f" % cv_testing_rmse(clf, data_v4, Agr)
    rmse_base['Agr'] = "%.3f" % rmse_mean(Agr)
    rmse_tr['Agr'] = "%.3f" % cv_training_rmse(clf, data_v4, Agr)

    rmse_pred['Con'] = "%.3f" % cv_testing_rmse(clf, data_v4, Con)
    rmse_base['Con'] = "%.3f" % rmse_mean(Con)
    rmse_tr['Con'] = "%.3f" % cv_training_rmse(clf, data_v4, Con)

    rmse_pred['Ope'] = "%.3f" % cv_testing_rmse(clf, data_v4, Ope)
    rmse_base['Ope'] = "%.3f" % rmse_mean(Ope)
    rmse_tr['Ope'] = "%.3f" % cv_training_rmse(clf, data_v4, Ope)

    print("rmse of data_v4 in testing")
    for pr in pr_list:
        print(rmse_pred[pr])


    print("")
    print("rmse of data_v4 training")
    for pr in pr_list:
        print(rmse_tr[pr])


    print("")
    print("rmse of mean of pr")
    for pr in pr_list:
        print(rmse_base[pr])
    '''

    #HH, Emo, Ext, Agr, Con, Ope
    '''
    clf = LogisticRegression()
    rmse_pred['HH'] = "%.3f" % cv_testing_rmse(clf, data_v5, HH)
    rmse_base['HH'] = "%.3f" % rmse_mean(HH)
    rmse_tr['HH'] = "%.3f" % cv_training_rmse(clf, data_v5, HH)

    rmse_pred['Emo'] = "%.3f" % cv_testing_rmse(clf, data_v5, Emo)
    rmse_base['Emo'] = "%.3f" % rmse_mean(Emo)
    rmse_tr['Emo'] = "%.3f" % cv_training_rmse(clf, data_v5, Emo)

    rmse_pred['Ext'] = "%.3f" % cv_testing_rmse(clf, data_v5, Ext)
    rmse_base['Ext'] = "%.3f" % rmse_mean(Ext)
    rmse_tr['Ext'] = "%.3f" % cv_training_rmse(clf, data_v5, Ext)

    rmse_pred['Agr'] = "%.3f" % cv_testing_rmse(clf, data_v5, Agr)
    rmse_base['Agr'] = "%.3f" % rmse_mean(Agr)
    rmse_tr['Agr'] = "%.3f" % cv_training_rmse(clf, data_v5, Agr)

    rmse_pred['Con'] = "%.3f" % cv_testing_rmse(clf, data_v5, Con)
    rmse_base['Con'] = "%.3f" % rmse_mean(Con)
    rmse_tr['Con'] = "%.3f" % cv_training_rmse(clf, data_v5, Con)

    rmse_pred['Ope'] = "%.3f" % cv_testing_rmse(clf, data_v5, Ope)
    rmse_base['Ope'] = "%.3f" % rmse_mean(Ope)
    rmse_tr['Ope'] = "%.3f" % cv_training_rmse(clf, data_v5, Ope)

    print("rmse of data_v5 in testing")
    for pr in pr_list:
        print(rmse_pred[pr])

    print("")
    print("rmse of data_v5 training")
    for pr in pr_list:
        print(rmse_tr[pr])

    print("")
    print("rmse of mean of pr")
    for pr in pr_list:
        print(rmse_base[pr])
    '''

    #HH, Emo, Ext, Agr, Con, Ope
    for alpha_val in range(1, 130, 1):
        print('alpha_val = ' + str(alpha_val*0.1))
        clf = Ridge(alpha = alpha_val*0.1)
        rmse_pred['HH'].append("%.3f" % cv_testing_rmse(clf, data_all, HH))
        rmse_base['HH'].append(5.799)
        rmse_tr['HH'].append("%.3f" % cv_training_rmse(clf, data_all, HH))

        rmse_pred['Emo'].append("%.3f" % cv_testing_rmse(clf, data_all, Emo))
        rmse_base['Emo'].append(5.769)
        rmse_tr['Emo'].append("%.3f" % cv_training_rmse(clf, data_all, Emo))

        rmse_pred['Ext'].append("%.3f" % cv_testing_rmse(clf, data_all, Ext))
        rmse_base['Ext'].append(5.743)
        rmse_tr['Ext'].append("%.3f" % cv_training_rmse(clf, data_all, Ext))

        rmse_pred['Agr'].append("%.3f" % cv_testing_rmse(clf, data_all, Agr))
        rmse_base['Agr'].append(5.608)
        rmse_tr['Agr'].append("%.3f" % cv_training_rmse(clf, data_all, Agr))

        rmse_pred['Con'].append("%.3f" % cv_testing_rmse(clf, data_all, Con))
        rmse_base['Con'].append(5.239)
        rmse_tr['Con'].append("%.3f" % cv_training_rmse(clf, data_all, Con))

        rmse_pred['Ope'].append("%.3f" % cv_testing_rmse(clf, data_all, Ope))
        rmse_base['Ope'].append(5.355)
        rmse_tr['Ope'].append("%.3f" % cv_training_rmse(clf, data_all, Ope))

    # print("rmse of data_all in testing")
    # for pr in pr_list:
    #     print(rmse_pred[pr])
    #
    # print("")
    # print("rmse of data_all training")
    # for pr in pr_list:
    #     print(rmse_tr[pr])
    #
    # print("")
    # print("rmse of mean of pr")
    # for pr in pr_list:
    #     print(rmse_base[pr])

    #plot
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


if __name__ == '__main__':
    main()
