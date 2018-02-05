import csv
import pymysql
import pandas as pd

def connection():
    connection = pymysql.connect(host= "140.115.59.250",
                                 port=3307,
                                 user= "people400",
                                 password= "dart01",
                                 db='people400',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor
                                 )

    cursor = connection.cursor()
    sql = "SELECT * FROM `user_per`"
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        return result

    except:
        print("fetch error")

    connection.close()

def load_user_list():
    user_list = pd.read_csv("user_list.csv")
    return user_list

def choose_user():
    usernum = []
    user_pr = connection()
    user_id_list = load_user_list()
    new_user_pr = []
    for i in range(len(user_id_list)):
        for user_pr_num in range(len(user_pr)):
            if user_pr[user_pr_num]["id"] == user_id_list["id"][i]:
                new_user_pr.append(user_pr[user_pr_num])
            else:
                pass

    return new_user_pr

def main():
    user_pr = choose_user()
    with open("user_pr.csv", "w") as fout:
        wr = csv.writer(fout)
        title = ["id", "HH", "Emo", "Ext", "Agr", "Con", "Ope"]
        wr.writerow(title)
        for us_pr_num in range(len(user_pr)):
            value = []
            for item in title:
                value.append(user_pr[us_pr_num][item])
            wr.writerow(value)


if __name__ == '__main__':
    main()
