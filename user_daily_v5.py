import operator
import csv
import pandas as pd
from datetime import date
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

def create_user():
    tmp = []
    cate_list = load_cate_list()
    tmp_dict = {}

    for cate_i in cate_list:
        tmp_dict.update({cate_i:0})

    return tmp_dict

def main():
    #variable
    fname = []
    daily = []
    user_count = 600
    cate_list_form = load_cate_list()
    tmp_user = []
    daily_count = []
    cate_list = {}
    for i in range(len(cate_list_form)):
        cate_list.update({cate_list_form[i]:i})

    #main
    #create file list
    for i in range(7, 8, 1):
        fname.append("all_user_csv_out_v2_" + str(i) + ".csv")

    #in each file
    for i in range(len(fname)):
        print("File "+ str(i+1))
        print("csv loading")
        data = load_csv(fname[i])
        now_id = 0
        print("loading complete")

        for j in range(len(data)):
            if j == 0:
                now_id = data["id"][j]
                user_count += 1
                #create new tmp_user
                user = create_user()

            else:
                pass

            if now_id != data["id"][j]:
                print("user: "+ str(user_count) + " complete")
                #output csv
                with open("user" + str(user_count) + "_daily.csv", "w") as fout:
                    wr = csv.writer(fout)
                    title = ["id"]
                    for cate_num_2 in cate_list_form:
                        title.append(cate_num_2)

                    wr.writerow(title)

                    value = []
                    value.append(now_id)
                    tmp_cate_value_1 = 0
                    tmp_cate_value_1 = sum(user.values())

                    if tmp_cate_value_1 == 0:
                        tmp_cate_value_1 = 1.0

                    else:
                        tmp_cate_value_1 = tmp_cate_value_1 * 1.0

                    for cate_num_3 in cate_list_form:
                        tmp_value = user[cate_num_3] / tmp_cate_value_1
                        value.append("%.3f" % tmp_value)

                    wr.writerow(value)

                now_id = data["id"][j]


                #create new tmp_user
                user = create_user()

                user_count += 1

            elif now_id == data["id"][j]:
                print(str(j) + "/" + str(len(data)))
                user[data["cate"][j]] += 1


            else:
                pass

if __name__ == "__main__":
    main()
