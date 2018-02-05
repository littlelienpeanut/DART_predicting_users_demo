import pandas as pd

def main():
    pr_but_not_demo = []
    demo_but_not_pr = []

    user_pr = pd.read_csv("user_pr.csv")
    user_demo = pd.read_csv("user_demo.csv")

    for pr_num in range(len(user_pr)):
        flag = 0
        for demo_num in range(len(user_demo)):
            if user_pr["id"][pr_num] == user_demo["id"][demo_num]:
                flag = 1

            else:
                pass
                
        if flag == 0:
            pr_but_not_demo.append(user_pr["id"][pr_num])

        else:
            pass


    for demo_num in range(len(user_demo)):
        flag = 0
        for pr_num in range(len(user_pr)):
            if user_demo["id"][demo_num] == user_pr["id"][pr_num]:
                flag = 1

            else:
                pass

        if flag == 0:
            demo_but_not_pr.append(user_demo["id"][demo_num])
        else:
            pass


    print("pr user but not demo: ")
    print(pr_but_not_demo)
    print("")
    print("demo user but not pr: ")
    print(demo_but_not_pr)

if __name__ == '__main__':
    main()
