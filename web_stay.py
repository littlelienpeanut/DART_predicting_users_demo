import time
import datetime
from datetime import datetime, timedelta
import csv
import operator
import pandas as pd
import numpy as np
from datetime import date

def load_csv(fname):
    csv = pd.read_csv(fname)
    return csv

def time_convert(input_time):
    time_con = datetime.strptime(input_time, "%Y-%m-%d-%H-%M-%S")
    return time_con

def main():
    #variable
    now_id = 0
    data_out = []
    #time slot from 5sec ~ 20min
    time_slot = []
    for i in range(0, 1205, 5):
        time_slot.append(i)


    #main
    for i in range(1, 8, 1):

        fname = "all_user_csv_out_v2_" + str(i) + ".csv"
        data = load_csv(fname)

        ftitle = "Youtube"
        target_url = "www.youtube.com"
        #target_url_2 = "www.google.com.tw"
        #target_url_3 = "mail.google.com"
        #target_url_4 = "www.youtube.com"

        for j in range(len(data)):
            print("file: " + str(i) + " " + str(j) + "/" + str(len(data)))
            if now_id != data["id"][j]:
                if j != 0:
                    tmp_0 = tdict[0]
                    tdict.pop(0, None)
                    usu_stay_time = max(tdict.iteritems(), key=operator.itemgetter(1))[0]
                    tdict.update({"usu_stay_time":usu_stay_time})
                    tdict.update({0:tmp_0})
                    tdict.update({"id":now_id})
                    data_out.append(tdict)



                else:
                    pass

                tdict = {}
                now_id = data["id"][j]
                for time in time_slot:
                    tdict.update({time:0})

            else:
                pass

            if data["cate"][j] == target_url:
                now_time = time_convert(data["visit_time"][j])
                for next_history in range(len(data)):
                    if j+next_history >= len(data):
                        break

                    elif data["visit_time"][j+next_history] == data["visit_time"][j]:
                        pass

                    else:
                        next_time = time_convert(data["visit_time"][j+next_history])
                        stay_time = (next_time - now_time).seconds

                        if stay_time > 1200:
                            pass

                        else:
                            stay_time = stay_time - (stay_time % 5)
                            tdict[stay_time] = tdict[stay_time] + 1
                            break


            else:
                pass


            if len(data) == j+1:
                tmp_0 = tdict[0]
                tdict.pop(0, None)
                usu_stay_time = max(tdict.iteritems(), key=operator.itemgetter(1))[0]
                tdict.update({"usu_stay_time":usu_stay_time})
                tdict.update({0:tmp_0})
                tdict.update({"id":now_id})
                data_out.append(tdict)

            else:
                pass

    with open("web_stay_" + ftitle + ".csv","w") as fout:
        wr = csv.writer(fout)
        title = ["id", "usu_stay_time"]
        for j in time_slot:
            title.append(j)

        wr.writerow(title)
        for k in range(len(data_out)):
            value = []
            value.append(data_out[k]["id"])
            value.append(data_out[k]["usu_stay_time"])
            for j in time_slot:
                value.append(data_out[k][j])
            wr.writerow(value)

if __name__ == '__main__':
    main()
