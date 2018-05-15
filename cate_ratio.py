import pandas as pd
import csv


def new_day(cate_list, day, date):
    date.update({day:{}})
    for cate in cate_list:
        date[day].update({cate:0})

def load_cate_list():
    list = []
    csv = pd.read_csv("cate_list_new.csv")
    for i in range(len(csv)):
        list.append(csv["cate"][i])
    return list


def main():
    fname = []
    cate_list = load_cate_list()
    date = {}

    for data_num in range(1, 8, 1):
        fname.append('all_user_csv_out_v3_' + str(data_num) + '.csv')

    for fn in range(len(fname)):
        df = pd.read_csv(fname[fn])
        for row in range(len(df)):
            print('file: ' + str(fn+1) + ' ' + str(row) + '/' + str(len(df)))
            if df['visit_time'][row][:10] in date.keys():
                date[df['visit_time'][row][:10]][df['cate'][row]] += 1

            else:
                new_day(cate_list, df['visit_time'][row][:10], date)
                date[df['visit_time'][row][:10]][df['cate'][row]] += 1

    with open('cate_day_ratio.csv', 'w', newline='') as fout:
        wr = csv.writer(fout)
        title = ['date']
        title.extend(cate_list)
        wr.writerow(title)

        for key, value in date.items():
            #key: date; value: dict{cate:num}
            row_value = []
            row_value.append(key)
            for cate in cate_list:
                row_value.append(value[cate])

            wr.writerow(row_value)








if __name__ == '__main__':
    main()
