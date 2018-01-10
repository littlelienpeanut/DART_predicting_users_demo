import operator
import csv
import pandas as pd
from datetime import date

def time2sec(s):
    s = s[11:]
    hour, minute, second  =  s.split('-')
    total = int(hour)*60*60 + int(minute)*60 + int(second)
    return  int(total)

def load_csv(fname):
    csv = pd.read_csv(fname)
    for i in range(len(csv)):
        print(str(i) + "/" + str(len(csv)))
        csv["visit_time"][i] = time2sec(csv["visit_time"][i])

    return csv

def load_cate_list():
    list = []
    csv = pd.read_csv("cate_list_final.csv")
    for i in range(len(csv)):
        list.append(csv["cate"][i])
    return list

def sec2time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def main():
    #variable
    fname = []
    daily = []
    user_count = 0
    cate_list_form = load_cate_list()

    #main
    #create file list
    for i in range(1, 2, 1):
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
                daily = []
                for d_i in range(17280):
                    tdict = {}
                    tdict.update({"count":0})
                    for cate_num_1 in cate_list_form:
                        tdict.update({cate_num_1:0})
                    daily.append(tdict)

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

                    for wr_i in range(len(daily)):
                        value = []
                        value.append(now_id)
                        value.append(sec2time(wr_i*5))

                        if daily[wr_i]["count"] == 0:
                            daily[wr_i]["count"] = -1
                            tmp_count = -1.0

                        else:
                            tmp_count = daily[wr_i]["count"] * 1.0
                            del daily[wr_i]["count"]


                        for cate_num_3 in cate_list_form:
                            tmp_cate_value_1 = daily[wr_i][cate_num_3]*1.0
                            tmp_cate_value_2 = tmp_cate_value_1/tmp_count
                            value.append("%.3f" % tmp_cate_value_2)

                        wr.writerow(value)

                now_id = data["id"][j]
                daily = []
                for d_i in range(17280):
                    tdict = {}
                    tdict.update({"count":0})
                    for cate_num in cate_list_form:
                        tdict.update({cate_num:0})
                    daily.append(tdict)

                user_count += 1

            elif now_id == data["id"][j]:
                now_time = int(data["visit_time"][j]/5)
                if data["cate"][j] in daily[now_time]:
                    daily[now_time][data["cate"][j]] += 1
                    daily[now_time]["count"] += 1

                else:
                    pass




if __name__ == "__main__":
    main()
