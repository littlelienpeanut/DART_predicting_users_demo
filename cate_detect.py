import operator
import csv
import pandas as pd

def load_csv(fname):
    csv = pd.read_csv(fname)
    return csv

def main():
    #variable
    fname = []
    cate_list = []
    user_count = 0

    #main
    #create file list
    for i in range(1, 8, 1):
        fname.append("cate_list_" + str(i) + ".csv")

    #in each file
    for i in range(len(fname)):
        print("File "+ str(i+1))
        print("csv loading")
        data = load_csv(fname[i])
        print("loading complete")

        for j in range(len(data)):
            print(str(j) + "/" + str(len(data)) + " in file: " + str(i+1))
            if data["cate"][j] in cate_list:
                pass

            else:
                print("new cate: " + str(data["cate"][j]))
                cate_list.append(str(data["cate"][j]))

    with open("cate_list_final.csv", "w") as fout:
        wr = csv.writer(fout)
        title = ["cate"]
        wr.writerow(title)

        for i in range(len(cate_list)):
            value = []
            value.append(cate_list[i])
            wr.writerow(value)

if __name__ == '__main__':
    main()
