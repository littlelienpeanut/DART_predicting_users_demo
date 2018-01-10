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

def sec2time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def main():
    #variable
    fname = []
    daily = []
    user_count = 0

    #main
    #create file list
    for i in range(1, 3, 1):
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
                    tdict.update({"no_record_1":0})
                    tdict.update({"no_record_2":0})
                    tdict.update({"no_record_3":0})
                    daily.append(tdict)

            else:
                pass

            if now_id != data["id"][j]:
                print("user: "+ str(user_count) + " complete")
                #output csv
                with open("user" + str(user_count) + "_daily.csv", "w") as fout:
                    wr = csv.writer(fout)
                    title = ["id", "visit_time", "1st", "1st_prop", "2nd", "2nd_prop", "3rd", "3rd_prop"]
                    wr.writerow(title)

                    for wr_i in range(len(daily)):
                        value = []
                        value.append(now_id)
                        value.append(sec2time(wr_i*5))

                        #avoid no record
                        if daily[wr_i]["count"] == 0:
                            daily[wr_i]["count"] = -1
                            tmp_count = -1.0

                        else:
                            tmp_count = daily[wr_i]["count"] * 1.0
                            del daily[wr_i]["count"]

                        #1st
                        tmp_cate = max(daily[wr_i].iteritems(), key=operator.itemgetter(1))[0]
                        value.append(tmp_cate)
                        tmp_prop = daily[wr_i][tmp_cate]/tmp_count
                        value.append("%.3f" % tmp_prop)
                        del daily[wr_i][tmp_cate]

                        #2nd
                        tmp_cate = max(daily[wr_i].iteritems(), key=operator.itemgetter(1))[0]
                        value.append(tmp_cate)
                        tmp_prop = daily[wr_i][tmp_cate]/tmp_count
                        value.append("%.3f" % tmp_prop)
                        del daily[wr_i][tmp_cate]

                        #3rd
                        tmp_cate = max(daily[wr_i].iteritems(), key=operator.itemgetter(1))[0]
                        value.append(tmp_cate)
                        tmp_prop = daily[wr_i][tmp_cate]/tmp_count
                        value.append("%.3f" % tmp_prop)
                        del daily[wr_i][tmp_cate]

                        wr.writerow(value)

                now_id = data["id"][j]
                daily = []
                for d_i in range(17280):
                    tdict = {}
                    tdict.update({"count":0})
                    tdict.update({"no_record_1":0})
                    tdict.update({"no_record_2":0})
                    tdict.update({"no_record_3":0})
                    daily.append(tdict)

                user_count += 1

            elif now_id == data["id"][j]:
                now_time = data["visit_time"][j]/5

                if data["cate"][j] in daily[now_time]:
                    daily[now_time][data["cate"][j]] += 1
                    daily[now_time]["count"] += 1

                else:
                    daily[now_time].update({data["cate"][j]:1})
                    daily[now_time]["count"] += 1




if __name__ == "__main__":
    main()
