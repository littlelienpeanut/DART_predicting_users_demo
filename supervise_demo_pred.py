import pandas as pd
import csv
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np


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

def load_user_demo():
    user_demo = {}
    data = pd.read_csv("user_demo.csv")
    for i in range(len(data)):
        tdict = {}
        tdict.update({"age": data["age"][i]})
        tdict.update({"gender": data["gender"][i]})
        tdict.update({"relationship": data["relationship"][i]})
        tdict.update({"income": data["income"][i]})
        tdict.update({"edu": data["edu"][i]})
        tdict.update({"location": data["location"][i]})
        tdict.update({"occupation": data["occupation"][i]})
        tdict.update({"industry": data["industry"][i]})
        user_demo.update({data["id"][i]:tdict})
    return user_demo

def load_user_label():
    user_label_age = []
    user_label_gender = []
    user_label_relationship = []
    user_label_income = []
    user_label_edu = []
    user_label_location = []
    user_label_occupation = []
    user_label_industry = []
    user_id_list = load_user_list()
    user_demo = load_user_demo()
    for i in range(len(user_id_list)):
        try:
            user_label_age.append(user_demo[user_id_list["id"][i]]["age"])
            user_label_gender.append(user_demo[user_id_list["id"][i]]["gender"])
            user_label_relationship.append(user_demo[user_id_list["id"][i]]["relationship"])
            user_label_income.append(user_demo[user_id_list["id"][i]]["income"])
            user_label_edu.append(user_demo[user_id_list["id"][i]]["edu"])
            user_label_location.append(user_demo[user_id_list["id"][i]]["location"])
            user_label_occupation.append(user_demo[user_id_list["id"][i]]["occupation"])
            user_label_industry.append(user_demo[user_id_list["id"][i]]["industry"])

        except:
            pass

    return user_label_age, user_label_gender, user_label_relationship, user_label_income, user_label_edu, user_label_location, user_label_occupation, user_label_industry

def choose_user():
    usernum = []
    user_demo = load_user_demo()
    user_id_list = load_user_list()
    for i in range(len(user_id_list)):
        try:
            if user_demo[user_id_list["id"][i]]["age"]:
                usernum.append("user" + str(i+1))

        except:
            pass

    return usernum

def get_class_num(te_y):
    tmp_list = []
    for i in te_y:
        if i in tmp_list:
            pass
        else:
            tmp_list.append(i)

    return tmp_list

def main():
    #variable
    data_v4 = []
    data_v5 = []
    user_label = []
    usernum = []
    user_label_age = []
    user_label_gender = []
    user_label_relationship = []
    user_label_income = []
    user_label_edu = []
    user_label_location = []
    user_label_occupation = []
    user_label_industry = []
    label_name_list = ["age", "gender", "relationship", "income", "edu"]
    #data[0].id == user_id_list["id"][0]
    user_id_list = load_user_list()
    user_demo = load_user_demo()
    #index is user_i-1
    usernum = choose_user()
    data_v4 = load_data_v4(usernum)
    data_v5 = load_data_v5(usernum)
    user_label_age, user_label_gender, user_label_relationship, user_label_income, user_label_edu, user_label_location, user_label_occupation, user_label_industry = load_user_label()



    #main

    #find the best k in v4, v5 with every demographic
    #in v4 age: k=16, gender: k=15, relationship: k=1or8, income: k=4, edu: k=4
    v4_k = [16, 15, 1, 4, 4]
    #in v5 age: k=9, gender: k=1or3, relationship: k=5, income: k=7, edu: k=12
    v5_k = [9, 1, 5, 7, 12]
    """
    k_score = []
    k_range = range(1,20)
    for k in k_range:
        knn_v4_age = KNeighborsClassifier(n_neighbors = k)
        #knn_v4_age_scores = cross_val_score(knn_v4_age, data_v4, user_label_age, cv=5, scoring = "accuracy")
        tmp_score = []
        for iter in range(5):
            tr_x, te_x, tr_y, te_y = train_test_split(data_v5, user_label_edu, test_size = 0.2, random_state=42)
            knn_v4_age.fit(tr_x, tr_y)
            tmp_score.append(knn_v4_age.score(te_x, te_y))

        k_score.append(sum(tmp_score)/5.0)

    for i in range(len(k_score)):
        print("k = " + str(i+1) + " accuracy = " + str(k_score[i]))
    """

    #knn
    #age
    class_num = []
    knn_v4_age = KNeighborsClassifier(n_neighbors = 16)
    tr_x, te_x, tr_y, te_y = train_test_split(data_v4, user_label_age, test_size = 0.2, random_state=42)
    class_num = get_class_num(te_y)
    knn_v4_age.fit(tr_x, tr_y)
    print(class_num)
    y_score = knn_v4_age.predict_proba(te_x)
    y_score = np.array(y_score[:,1])
    te_y = np.asarray(te_y)
    knn_v4_age_fpr = dict()
    knn_v4_age_tpr = dict()
    knn_v4_age_roc_auc = dict()
    for i in range(class_num):
        knn_v4_age_fpr[i], knn_v4_age_tpr[i], _ = roc_curve(te_y[:, i], y_score[:, i])
        knn_v4_age_roc_auc[i] = auc(knn_v4_age_fpr[i], knn_v4_age_tpr[i])
    print(knn_v4_age_fpr)
    print(knn_v4_age_tpr)



if __name__ == '__main__':
    main()
