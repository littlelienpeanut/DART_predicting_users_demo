import pandas as pd
import csv
import math

def load_csv(fname):
    csv = pd.read_csv(fname)
    return csv

def load_cate_list():
    list = []
    csv = pd.read_csv("cate_list_final.csv")
    for i in range(len(csv)):
        list.append(csv["cate"][i])
    return list

def main():
    #variable
    user_1 = []
    user_2 = []
    user_similarity = []
    cate_list = load_cate_list

        #main
        for uid_1 in range(1, 2, 1):
            print("Loading " + str(uid_1) + " csv")
            user_1 = load_csv("user"+str(uid_1)+"_daily.csv")

            for uid_2 in range(1, 677, 1):
                print("Loading " + str(uid_2) + " csv")
                user_2 = load_csv("user"+str(uid_2)+"_daily.csv")
                similarity = []
                similarity_value = 0

                for row in range(1, 17280, 1):
                    tmp_value = 0
                    for cate in cate_list:
                        tmp_value += math.pow(user_1[cate][row] - user_2[cate][row], 2)
                    similarity.append(math.sqrt(tmp_value))


                similarity_value = sum(similarity)
                similarity_value = similarity_value/17280.0
                user_similarity.append("%.3f" % similarity_value)

            with open("user" + str(uid_1) + "_similarity.csv", "w") as fout:
                wr = csv.writer(fout)
                title = ["user", "similarity_value"]
                wr.writerow(title)
                for i in range(len(user_similarity)):
                    value = []
                    value.append("user"+str(i+1))
                    value.append(user_similarity[i])
                    wr.writerow(value)



    if __name__ == '__main__':
        main()
