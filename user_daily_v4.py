import operator
import csv
import pandas as pd
from datetime import date
import math

def load_csv(fname):
    csv = pd.read_csv(fname)

    return csv

def create_tmp_user():
    tmp_user = []

    for i in range(24):
        tmp_user.append(0)

    return tmp_user

def create_daily():
    daily = []

    for i in range(24):
        daily.append([])

    return daily

def main():
    #variable
    fname = []
    daily = []
    user_count = 600
    tmp_user = []
    daily_count = []

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
                    title = ["id", "visit_time", "ratio"]
                    wr.writerow(title)

                    for wr_i in range(24):
                        value = []
                        value.append(now_id)
                        value.append(wr_i)
                        if len(daily[wr_i]) == 0:
                            tmp_cate_value_1 = 0
                        else:
                            tmp_cate_value_1 = sum(daily[wr_i]) / (len(daily[wr_i])*1.0)

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
                    tmp_user[int(data["visit_time"][j][11:13])] += 1
                    count_day += 1

                elif now_day != data["visit_time"][j][0:10]:
                    #check day_history numbers
                    if count_day < 30:
                        tmp_user = create_tmp_user()
                        now_day = data["visit_time"][j][0:10]
                        count_day = 0
                        tmp_user[int(data["visit_time"][j][11:13])] += 1
                        count_day += 1

                    elif count_day >= 30:
                        #combine day to daily
                        for daily_i in range(24):
                            if(tmp_user[daily_i] >= 10):
                                daily[daily_i].append(1)
                            else:
                                daily[daily_i].append(0)
                        tmp_user = create_tmp_user()
                        now_day = data["visit_time"][j][0:10]
                        count_day = 0
                        tmp_user[int(data["visit_time"][j][11:13])] += 1
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
