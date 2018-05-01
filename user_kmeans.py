import pandas as pd
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
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

def plot_clustering(n_clusters, cluster_labels, X):
    plt.figure()
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    plt.set_title("The silhouette plot for the various clusters.")
    plt.set_xlabel("The silhouette coefficient values")
    plt.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.set_yticks([])  # Clear the yaxis labels / ticks
    plt.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()


def main():
    #variable
    data_v4 = []
    data_v5 = []
    kmeans_v4 = []
    kmeans_v5 = []
    kmeans_mix = []
    v4_score = []
    v5_score = []
    mix_score = []
    user_id_list = load_user_list()
    user_demo = load_user_demo()

    #main
    #index is user_i-1
    data_v4 = load_data_v4()
    data_v5 = load_data_v5()

    #daily_v4 kmeans
    for km4_i in range(0, 672, 1):
        kmeans_v4.append(data_v4[km4_i])

    #daily_v5 kmeans
    for km5_i in range(0, 672, 1):
        kmeans_v5.append(data_v5[km5_i])

    #daily_mix kmeans
    tmp_data_v4 = data_v4
    tmp_data_v5 = data_v5


    for kmm_i in range(0, 672, 1):
        tmp_list = []
        tmp_list = data_v4[kmm_i] + data_v5[kmm_i]
        kmeans_mix.append(tmp_list)

    for c_num in range(2, 20, 1):
        kmeans_v4_label = KMeans(n_clusters = c_num, random_state=2018).fit_predict(kmeans_v4)
        kms_v4_savg = silhouette_score(kmeans_v4, kmeans_v4_label)

        kmeans_v5_label = KMeans(n_clusters = c_num, random_state=2018).fit_predict(kmeans_v5)
        kms_v5_savg = silhouette_score(kmeans_v5, kmeans_v5_label)

        kmeans_mix_label = KMeans(n_clusters = c_num, random_state=2018).fit_predict(kmeans_mix)
        kms_mix_savg = silhouette_score(kmeans_mix, kmeans_mix_label)


        v4_score.append(kms_v4_savg)
        v5_score.append(kms_v5_savg)
        mix_score.append(kms_mix_savg)



    #print silhouette score
    for i in range(len(v4_score)):
        print('cnum= ' + str(i+2) + ' score: '+ str(v4_score[i]))
    print('')

    for i in range(len(v5_score)):
        print('cnum= ' + str(i+2) + ' score: '+ str(v5_score[i]))
    print('')

    for i in range(len(mix_score)):
        print('cnum= ' + str(i+2) + ' score: '+ str(mix_score[i]))

    #plot silhouette score
    k = np.arange(18)
    x_stick = list(range(2, 21, 1))
    plt.figure()
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    plt.plot(k, mix_score, 'r-')
    plt.xticks(k, x_stick)
    plt.ylabel("silhouette score")
    plt.xlabel("number of k in k-means")
    #plt.savefig("silhouette score.eps", format='eps', dpi=1000)
    plt.savefig("silhouette score.png", format='png', dpi=1000)
    #plt.show()


if __name__ == '__main__':
    main()
