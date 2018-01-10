import csv
import pandas as pd

def main():
    id_list = []

    for i in range(1, 673, 1):
        file_name = "user" + str(i) + "_daily_v5.csv"
        fin = pd.read_csv(file_name)
        id_list.append(fin["id"][0])

    with open("user_list.csv", "w") as fout:
        wr = csv.writer(fout)
        title = ["id"]
        wr.writerow(title)
        for id in id_list:
            value = []
            value.append(id)
            wr.writerow(value)

if __name__ == '__main__':
    main()
