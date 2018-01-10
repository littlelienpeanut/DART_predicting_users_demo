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

def create_tmp_user():
    tmp = []
    cate_list = load_cate_list()

    for i in range(24):
        cate = []
        for i in range(len(cate_list)):
            cate.append(0)
        tmp.append(cate)

    return tmp

def create_daily():
    tmp = []
    cate_list = load_cate_list()

    for i in range(24):
        cate = {}
        for c in cate_list:
            cate.update({c: []})
        tmp.append(cate)

    return tmp

def main():
    #variable
    fname = []
    daily = []
    user_count = 6
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
                tmp_user = create_tmp_user()
                daily = create_daily()
                now_day = data["visit_time"][j][0:10]
                count_day = 0

            else:
                pass

            if now_id != data["id"][j]:
                print("user: "+ str(user_count) + " complete")
                #output csv
                with open("user" + str(user_count) + "_daily.csv", "w") as fout:
                    wr = csv.writer(fout)
                    title = ["id", "visit_time"]
                    for cate_num_2 in cate_list_form:
                        title.append(cate_num_2)

                    wr.writerow(title)
                    for wr_i2 in range(24):
                        tmp_count = 0
                        for cate_num_3 in cate_list_form:
                            tmp_count += sum(daily[wr_i2][cate_num_3])
                        if tmp_count == 0:
                            tmp_count = 1
                        else:
                            pass
                        daily_count.append(tmp_count)


                    for wr_i in range(len(daily)):
                        value = []
                        value.append(now_id)
                        value.append(wr_i)


                        for cate_num_3 in cate_list_form:
                            tmp_cate_value_1 = sum(daily[wr_i][cate_num_3]) / (daily_count[wr_i]*1.0)
                            value.append("%.3f" % tmp_cate_value_1)

                        wr.writerow(value)

                now_id = data["id"][j]
                daily = []
                daily_count = []

                #create new tmp_user
                tmp_user = create_tmp_user()
                daily = create_daily()
                now_day = 0
                count_day = 0

                user_count += 1

            elif now_id == data["id"][j]:
                print(str(j) + "/" + str(len(data)))
                #still today
                if now_day == data["visit_time"][j][0:10]:
                    tmp_value = tmp_user[int(data["visit_time"][j][11:13])][cate_list[data["cate"][j]]] + 1
                    tmp_user[int(data["visit_time"][j][11:13])][cate_list[data["cate"][j]]] = tmp_value
                    count_day += 1
                elif now_day != data["visit_time"][j][0:10]:
                    #check day_history numbers
                    if count_day < 30:
                        tmp_user = create_tmp_user()
                        now_day = data["visit_time"][j][0:10]
                        count_day = 0
                        if data["cate"][j] == "nan":
                            data["cate"][j] = data["cate"][j-1]
                        else:
                            pass

                        tmp_value = tmp_user[int(data["visit_time"][j][11:13])][cate_list[data["cate"][j]]] + 1
                        tmp_user[int(data["visit_time"][j][11:13])][cate_list[data["cate"][j]]] = tmp_value
                        count_day += 1

                    elif count_day >= 30:
                        #combine day to daily
                        for daily_i in range(24):
                            for cate_i in cate_list_form:
                                tmp_list_2 = []
                                tmp_list_2 = daily[daily_i][cate_i]
                                tmp_list_2.append(tmp_user[daily_i][cate_list[cate_i]])
                                daily[daily_i].update({cate_i: tmp_list_2})

                        tmp_user = create_tmp_user()
                        now_day = data["visit_time"][j][0:10]
                        count_day = 0
                        tmp_user[int(data["visit_time"][j][11:13])][cate_list[data["cate"][j]]] += 1
                        count_day += 1

                    else:
                        print("combine error")
                        input()

                else:
                    pass

            else:
                pass

if __name__ == "__main__":
    main()
