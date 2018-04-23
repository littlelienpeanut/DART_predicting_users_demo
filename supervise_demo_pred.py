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
from sklearn.ensemble import RandomForestClassifier
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

def choose_user(user_demo_id):
    #return the user_num:1~672 who has demographic data as "userN" and the user index in user_demo
    usernum = []
    user_demo = load_user_demo() #509
    user_id_list = load_user_list() #672
    for i in range(len(user_id_list)):
        try:
            #if user has user_demo
            if user_demo_id[user_id_list["id"][i]] == 0:
                usernum.append("user" + str(i+1))

        except:
            pass

    return usernum

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #compute overall accuracy
    oc = 0
    total = 0
    tmp_acc = 0
    for cm_i in range(len(cm)):
        tmp_acc += cm[cm_i][cm_i]
        for cm_j in range(len(cm)):
            total += cm[cm_i][cm_j]
    oc = float(tmp_acc) / float(total)

    tmp_recall = np.array(recall(cm))
    tmp_recall = tmp_recall.astype(float)
    tmp_precision =  np.array(precision(cm, oc))
    tmp_precision = tmp_precision.astype(float)
    cm_nn = cm
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    tmp_recall = tmp_recall[:, np.newaxis]
    cm = np.hstack((cm, tmp_recall))
    cm = np.vstack((cm, tmp_precision))

    class_x = []
    class_y = []

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes)+1)
    #add recall tick
    classes.append("recall")
    plt.xticks(tick_marks, classes, rotation=45)
    #delete recall tick
    classes.pop(len(classes)-1)
    #add precision tick
    classes.append("precision")
    plt.yticks(tick_marks, classes)
    #delete precision tick
    classes.pop(len(classes)-1)

    fmt = '2.1%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 color="black")
        if i<cm.shape[0]-1 and j<cm.shape[0]-1:
            plt.text(j, i, cm_nn[i, j],
                 weight = 'bold',
                 horizontalalignment='center',
                 verticalalignment='top',
                 color="black")
        else:
            pass

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def load_user_demo():
    user_demo = {'id':[], 'age':[], 'gender':[], 'relationship':[], 'income':[], 'edu':[]}
    data = pd.read_csv("user_demo.csv")
    user_demo_id = {}
    demo_list = ['id', 'age', 'gender', 'relationship', 'income', 'edu']

    for i in range(len(data)):
        for demo_i in demo_list:
            user_demo[demo_i].append(data[demo_i][i])

        user_demo_id.update({data["id"][i]:0})

    return user_demo, user_demo_id

def recall(cnf):
    recall = []
    for i in range(len(cnf[0])):
        tmp_recall = float(cnf[i][i]) / float(sum(cnf[i]))
        recall.append(tmp_recall)

    return recall

def precision(cnf, oc):
    precision = []
    for i in range(len(cnf[0])):
        tmp = 0
        pre = 0
        for j in range(len(cnf[0])):
            tmp += cnf[j][i]
        if tmp == 0:
            tmp += 1

        pre = float(cnf[i][i]) / float(tmp)
        precision.append(pre)
    precision.append(oc)

    return precision


def main():
    #variable
    ### user_demo[id]['demographic'] ###
    print("Loading users' info")
    user_demo = []
    user_demo_id = {}
    user_demo, user_demo_id = load_user_demo()
    usernum = choose_user(user_demo_id)
    data_v4 = load_data_v4(usernum)
    data_v5 = load_data_v5(usernum)
    data_all = load_data_all(data_v4, data_v5)
    demo_list = ['age', 'gender', 'relationship', 'income', 'edu']

    #class name
    class_name = {'age':["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"], 'gender':["1", "2", "3"], 'relationship':["1", "2", "3", "4", "5"], 'income':["1", "2", "3", "4", "5", "6", "7", "8"], 'edu':["1", "2", "3", "4", "5", "7"]}


    #main
    #which dataset?
    user_data = data_all

    # ------------------------------------------------------------------------#
    demo_pred_score = {'age':[], 'gender':[], 'relationship':[], 'income':[], 'edu':[]}

    for k in range(1, 10, 1):
        ### classifier choosing
        clf = RandomForestClassifier(max_depth=k, random_state=2018)
        #clf = svm.SVC(C = k*0.1, class_weight ='balanced')
        print('max_depth: ' + str(k))
        for demo_i in demo_list:
            '''
            demo_pred[str(c_num)][demo_i].append(cross_val_predict(clf, kms_data[str(c_num)], kms_demo[str(c_num)][demo_i], cv=5))
            cnf_matrix = confusion_matrix(kms_demo[str(c_num)][demo_i], demo_pred[str(c_num)][demo_i][k-2])
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_name[demo_i], normalize=True, title=demo_i + ' in k = ' + str(k))
            plt.show()
            '''
            #f1-micro and f1-macro
            demo_pred_score[demo_i].append(np.mean(cross_val_score(clf, user_data, user_demo[demo_i], cv=5, scoring='f1_micro')))


    #print the best f1-micro with k_value
    for demo_i in demo_list:
        print(demo_i + ' /   Best testing score: ' + str('%.3f' % max(demo_pred_score[demo_i])) + ' /  k : ' + str(demo_pred_score[demo_i].index(max(demo_pred_score[demo_i]))+2))
    print(demo_pred_score)


if __name__ == '__main__':
    main()
