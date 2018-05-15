import pandas as pd
import csv

def main():
    fname = []

    for data_num in range(1, 8, 1):
        fname.append('all_user_csv_out_v2_' + str(data_num) + '.csv')

    for fn in range(len(fname)):
        df = pd.read_csv(fname[fn])
        for raw in range(len(df)):
            print('file: ' + str(fn+1) + ' ' + str(raw) + '/' + str(len(df)))
            if df['cate'][raw] == 'Auction' or df['cate'][raw] == 'Shopping' or df['cate'][raw] == 'Shopping and Auction':
                df['cate'][raw] = 'Shopping'

            else:
                pass

        with open('all_user_csv_out_v3_' + str(fn+1) + '.csv', 'w', newline='') as fout:
            wr = csv.writer(fout)
            title = ['id', 'visit_time', 'cate']
            wr.writerow(title)

            for raw in range(len(df)):
                value = []
                value.append(df['id'][raw])
                value.append(df['visit_time'][raw])
                value.append(df['cate'][raw])
                wr.writerow(value)

if __name__ == '__main__':
    main()
