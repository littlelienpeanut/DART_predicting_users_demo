import csv

for i in range(1, 673, 1):
    print("file " + str(i))
    with open("user" + str(i) + "_daily.csv", "r") as fin:
        with open("user" + str(i) + "_daily_v4.csv", "w") as fout:
            wr = csv.writer(fout)
            for line in csv.reader(fin):
                print(line)
                if not line:
                    pass
                else:
                    wr.writerow(line)
