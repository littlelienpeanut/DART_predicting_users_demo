import pandas as pd
import csv

def load_bh(fname):
    data = pd.read_csv(fname)
    return data['cate']

def main():
    cate_list = pd.read_csv('cate_list_final.csv')
    cate_count = {}
    fname = []
    total_count = 0

    for cate_name in cate_list['cate']:
        cate_count.update({cate_name:0})

    for fname_i in range(1, 8, 1):
        fname.append('all_user_csv_out_v2_' + str(fname_i) + '.csv')

    for name in fname:
        data = load_bh(name)
        for row_i in range(len(data)):
            print(name + ' ' + str(row_i) + ' / ' + str(len(data)))
            cate_count[data[row_i]] += 1
            total_count += 1

    for cate_name in cate_list['cate']:
        cate_count[cate_name] = cate_count[cate_name] / float(total_count)

    print('total: ' + str(total_count))

    with open('cate_ratio.csv', 'w') as fout:
        wr = csv.writer(fout)
        wr.writerow(cate_list['cate'])

        value = []
        for cate_name in cate_list['cate']:
            value.append(cate_count[cate_name])

        wr.writerow(value)

if __name__ == '__main__':
    main()
