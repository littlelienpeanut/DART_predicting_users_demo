import pandas as pd
import csv

def load_csv(fname):
    csv = pd.read_csv(fname)
    return csv

def main():
    #variable
    user_1 = []
    user_2 = []
    user_similarity = []


    #main
    for uid_1 in range(1, 101, 1):
        print("Loading " + str(uid_1) + " csv")
        user_1 = load_csv("user"+str(uid_1)+"_daily.csv")
        user_similarity = []

        for uid_2 in range(1, 673, 1):
            print("Loading " + str(uid_2) + " csv")
            user_2 = load_csv("user"+str(uid_2)+"_daily.csv")
            similarity = []
            similarity_value = 0

            for row in range(1, 17280, 1):
                if user_1["1st"][row][:9] != "no_record":
                    if user_1["1st"][row] == user_2["1st"][row]:
                        numerator = "%.3f" % (min(user_1["1st_prop"][row], user_2["1st_prop"][row]) * 1.0)
                        denominator = "%.3f" % (max(user_1["1st_prop"][row], user_2["1st_prop"][row]) * 1.0)
                        similarity.append(float(numerator)/float(denominator))

                    elif user_1["1st"][row] == user_2["2nd"][row]:
                        numerator = "%.3f" % (min(user_1["1st_prop"][row], user_2["2nd_prop"][row]) * 1.0)
                        denominator = "%.3f" % (max(user_1["1st_prop"][row], user_2["2nd_prop"][row]) * 1.0)
                        similarity.append(float(numerator)/float(denominator))

                    elif user_1["1st"][row] == user_2["3rd"][row]:
                        numerator = "%.3f" % (min(user_1["1st_prop"][row], user_2["3rd_prop"][row]) * 1.0)
                        denominator = "%.3f" % (max(user_1["1st_prop"][row], user_2["3rd_prop"][row]) * 1.0)
                        similarity.append(float(numerator)/float(denominator))

                    else:
                        pass

                elif user_1["2nd"][row][:9] != "no_record":
                    if user_1["2nd"][row] == user_2["1st"][row]:
                        numerator = "%.3f" % (min(user_1["2nd_prop"][row], user_2["1st_prop"][row]) * 1.0)
                        denominator = "%.3f" % (max(user_1["2nd_prop"][row], user_2["1st_prop"][row]) * 1.0)
                        similarity.append(float(numerator)/float(denominator))

                    elif user_1["2nd"][row] == user_2["2nd"][row]:
                        numerator = "%.3f" % (min(user_1["2nd_prop"][row], user_2["2nd_prop"][row]) * 1.0)
                        denominator = "%.3f" % (max(user_1["2nd_prop"][row], user_2["2nd_prop"][row]) * 1.0)
                        similarity.append(float(numerator)/float(denominator))

                    elif user_1["2nd"][row] == user_2["3rd"][row]:
                        numerator = "%.3f" % (min(user_1["2nd_prop"][row], user_2["3rd_prop"][row]) * 1.0)
                        denominator = "%.3f" % (max(user_1["2nd_prop"][row], user_2["3rd_prop"][row]) * 1.0)
                        similarity.append(float(numerator)/float(denominator))

                    else:
                        pass

                elif user_1["3rd"][row][:9] != "no_record":
                    if user_1["3rd"][row] == user_2["1st"][row]:
                        numerator = "%.3f" % (min(user_1["3rd_prop"][row], user_2["1st_prop"][row]) * 1.0)
                        denominator = "%.3f" % (max(user_1["3rd_prop"][row], user_2["1st_prop"][row]) * 1.0)
                        similarity.append(float(numerator)/float(denominator))

                    elif user_1["3rd"][row] == user_2["2nd"][row]:
                        numerator = "%.3f" % (min(user_1["3rd_prop"][row], user_2["2nd_prop"][row]) * 1.0)
                        denominator = "%.3f" % (max(user_1["3rd_prop"][row], user_2["2nd_prop"][row]) * 1.0)
                        similarity.append(float(numerator)/float(denominator))

                    elif user_1["3rd"][row] == user_2["3rd"][row]:
                        numerator = "%.3f" % (min(user_1["3rd_prop"][row], user_2["3rd_prop"][row]) * 1.0)
                        denominator = "%.3f" % (max(user_1["3rd_prop"][row], user_2["3rd_prop"][row]) * 1.0)
                        similarity.append(float(numerator)/float(denominator))

                    else:
                        similarity.append(0)


                elif user_2["1st"][row] == "no_record_1":
                      similarity.append(1.0)

                else:
                    similarity.append(0)

            similarity_value = sum(similarity)
            similarity_value = similarity_value/17280.0
            user_similarity.append("%.3f" % similarity_value)

        with open("user" + str(uid_1) + "_similarity.csv", "w") as fout:
            wr = csv.writer(fout)
            title = ["user", "similarity_value"]
            wr.writerow(title)
            for i in range(len(user_similarity)):
                value = []
                value.append("user"+str(i+1))
                value.append(user_similarity[i])
                wr.writerow(value)



if __name__ == '__main__':
    main()
