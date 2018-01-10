import pandas as pd

def load_csv(fname):
    csv = pd.read_csv(fname)
    return csv

def main():

    now_id = 0
    count = 0

    for i in range(7, 8, 1):

        fname = "all_user_csv_out_v2_" + str(i) + ".csv"
        data = load_csv(fname)

        for j in range(len(data)):
            print(str(j) + "/" + str(len(data)))
            if now_id != data["id"][j]:
                count += 1
                now_id = data["id"][j]

            else:
                pass


    print("users number: " + str(count))

if __name__ == "__main__":
    main()
