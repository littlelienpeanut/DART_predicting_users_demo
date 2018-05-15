import pandas as pd
import csv

def load_cate_list():
    list = []
    csv = pd.read_csv("cate_list_new.csv")
    for i in range(len(csv)):
        list.append(csv["cate"][i])
    return list

def main():
    cate_ratio = []
    df = pd.read_csv('cate_ratio_8to12.csv')
    cate_list = load_cate_list()

    #which cate wanna calculate?
    cate_name = 'Travel'

    for row in range(len(df)):
        day_total_click = 0
        day_cate_click = 0
        tmp_ratio = 0
        for cate in cate_list:
            if cate == cate_name:
                day_cate_click = df[cate][row]

            day_total_click += df[cate][row]

        cate_ratio.append(day_cate_click/float(day_total_click))

    with open(cate_name + '_ratio.csv', 'w', newline='') as fout:
        wr = csv.writer(fout)
        title = ['date', 'ratio']
        wr.writerow(title)

        for row in range(len(cate_ratio)):
            value = []
            value.append(df['date'][row])
            value.append(cate_ratio[row])

            wr.writerow(value)





if __name__ == '__main__':
    main()
