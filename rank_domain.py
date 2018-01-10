import pandas as pd
import operator

def load_csv(fname):
    csv = pd.read_csv(fname)
    return csv

def main():
    domain_list = {}

    for i in range(1, 8, 1):
        fname = "all_user_csv_out_" + str(i) + ".csv"
        data = load_csv(fname)
        for j in range(len(data)):
            print(str(j) + "/" + str(len(data)))
            if data["domain"][j] in domain_list.keys():
                domain_list[data["domain"][j]] = domain_list[data["domain"][j]] + 1

            else:
                domain_list.update({data["domain"][j]:0})


    for i in range(10):
        tmp_domain = max(domain_list.iteritems(), key=operator.itemgetter(1))[0]
        print(tmp_domain)
        del domain_list[tmp_domain]


if __name__ == '__main__':
    main()
