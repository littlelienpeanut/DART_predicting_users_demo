import pandas as pd
import csv
from sklearn.cluster import KMeans
from sklearn import metrics

def load_cate_list():
    list = []
    csv = pd.read_csv("cate_list_final.csv")
    for i in range(len(csv)):
        list.append(csv["cate"][i])
    return list

def load_data_v4():
    daily_v4 = []

    for i in range(1, 673, 1):
        tmp_list = []
        data = pd.read_csv("user" + str(i) + "_daily_v4.csv")
        for j in range(0, 24, 1):
            tmp_list.append( "%.3f" % data["ratio"][j])
        daily_v4.append(tmp_list)

    return daily_v4

def load_data_v5():
    daily_v5 = []
    cate_list = load_cate_list()

    for i in range(1, 673, 1):
        tmp_list = []
        data = pd.read_csv("user" + str(i) + "_daily_v5.csv")
        for cate in cate_list:
            tmp_list.append( "%.3f" % data[cate][0])
        daily_v5.append(tmp_list)

    return daily_v5

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

def main():
    #variable
    data_v4 = []
    data_v5 = []
    kmeans_v4 = []
    kmeans_v5 = []
    kmeans_mix = []
    user_id_list = load_user_list()
    user_demo = load_user_demo()

    #main
    #index is user_i-1
    data_v4 = load_data_v4()
    data_v5 = load_data_v5()

    #daily_v4 kmeans
    for km4_i in range(0, 672, 1):
        kmeans_v4.append(data_v4[km4_i])

    kmeans_v4_model_c3 = KMeans(n_clusters = 3).fit(kmeans_v4)
    kmeans_v4_model_c4 = KMeans(n_clusters = 4).fit(kmeans_v4)
    kmeans_v4_model_c5 = KMeans(n_clusters = 5).fit(kmeans_v4)

    #daily_v5 kmeans
    for km5_i in range(0, 672, 1):
        kmeans_v5.append(data_v5[km5_i])

    kmeans_v5_model_c3 = KMeans(n_clusters = 3).fit(kmeans_v5)
    kmeans_v5_model_c4 = KMeans(n_clusters = 4).fit(kmeans_v5)
    kmeans_v5_model_c5 = KMeans(n_clusters = 5).fit(kmeans_v5)


    #daily_mix kmeans
    tmp_data_v4 = data_v4
    tmp_data_v5 = data_v5


    for kmm_i in range(0, 672, 1):
        tmp_list = []
        tmp_list = data_v4[kmm_i] + data_v5[kmm_i]
        kmeans_mix.append(tmp_list)

    kmeans_mix_model_c3 = KMeans(n_clusters = 3).fit(kmeans_mix)
    kmeans_mix_model_c4 = KMeans(n_clusters = 4).fit(kmeans_mix)
    kmeans_mix_model_c5 = KMeans(n_clusters = 5).fit(kmeans_mix)


    #others
    #print n_clusters=3 feature: user_daily_v4 result
    #id_index is user_id - 1
    km4_c3 = {"0":[], "1":[], "2":[]}
    for o_i in range(0, 672, 1):
        km4_c3[str(kmeans_v4_model_c3.labels_[o_i])].append(o_i)
    '''
    print("k4_c3_0 " + str(km4_c3["0"]))
    print("")
    print("k4_c3_1 " + str(km4_c3["1"]))
    print("")
    print("k4_c3_2 " + str(km4_c3["2"]))
    print("")
    print("")
    '''
    
    with open("cluster_0_demo.csv", "w") as fout:
        wr = csv.writer(fout)
        title = ["age", "gender", "relationship", "income", "edu", "location", "occupation", "industry"]
        wr.writerow(title)

        for user_index in km4_c3["0"]:
            try:
                value = []
                value.append(user_demo[user_id_list["id"][user_index+1]]["age"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["gender"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["relationship"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["income"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["edu"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["location"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["occupation"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["industry"])
                wr.writerow(value)
            except:
                pass
            #print("user: " + str(user_id_list["id"][user_index+1]) + " does not have demographic data.")

    with open("cluster_1_demo.csv", "w") as fout:
        wr = csv.writer(fout)
        title = ["age", "gender", "relationship", "income", "edu", "location", "occupation", "industry"]
        wr.writerow(title)

        for user_index in km4_c3["1"]:
            try:
                value = []
                value.append(user_demo[user_id_list["id"][user_index+1]]["age"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["gender"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["relationship"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["income"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["edu"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["location"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["occupation"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["industry"])
                wr.writerow(value)
            except:
                pass
            #print("user: " + str(user_id_list["id"][user_index+1]) + " does not have demographic data.")

    with open("cluster_2_demo.csv", "w") as fout:
        wr = csv.writer(fout)
        title = ["age", "gender", "relationship", "income", "edu", "location", "occupation", "industry"]
        wr.writerow(title)

        for user_index in km4_c3["2"]:
            try:
                value = []
                value.append(user_demo[user_id_list["id"][user_index+1]]["age"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["gender"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["relationship"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["income"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["edu"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["location"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["occupation"])
                value.append(user_demo[user_id_list["id"][user_index+1]]["industry"])
                wr.writerow(value)
            except:
                pass
            #print("user: " + str(user_id_list["id"][user_index+1]) + " does not have demographic data.")
    #print silhouette score
    silhouette_k4_c3 = metrics.silhouette_score(kmeans_v4, kmeans_v4_model_c3.labels_)
    print("k4_c3_score: " + str(silhouette_k4_c3))
    silhouette_k4_c4 = metrics.silhouette_score(kmeans_v4, kmeans_v4_model_c4.labels_)
    print("k4_c4_score: " + str(silhouette_k4_c4))
    silhouette_k4_c5 = metrics.silhouette_score(kmeans_v4, kmeans_v4_model_c5.labels_)
    print("k4_c5_score: " + str(silhouette_k4_c5))
    silhouette_k5_c3 = metrics.silhouette_score(kmeans_v5, kmeans_v5_model_c3.labels_)
    print("k5_c3_score: " + str(silhouette_k5_c3))
    silhouette_k5_c4 = metrics.silhouette_score(kmeans_v5, kmeans_v5_model_c4.labels_)
    print("k5_c4_score: " + str(silhouette_k5_c4))
    silhouette_k5_c5 = metrics.silhouette_score(kmeans_v5, kmeans_v5_model_c5.labels_)
    print("k5_c5_score: " + str(silhouette_k5_c5))
    silhouette_km_c3 = metrics.silhouette_score(kmeans_mix, kmeans_mix_model_c3.labels_)
    print("km_c3_score: " + str(silhouette_km_c3))
    silhouette_km_c4 = metrics.silhouette_score(kmeans_mix, kmeans_mix_model_c4.labels_)
    print("km_c4_score: " + str(silhouette_km_c4))
    silhouette_km_c5 = metrics.silhouette_score(kmeans_mix, kmeans_mix_model_c5.labels_)
    print("km_c5_score: " + str(silhouette_km_c5))



if __name__ == '__main__':
    main()
