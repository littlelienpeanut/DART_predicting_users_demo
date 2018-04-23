import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    data = pd.read_csv('pr_dist.csv')
    x_stick = ['0~5', '6~10', '11~15', '16~20', '21~25', '26~30', '31~35', '36~40', '41~45', '46~50']
    k = np.arange(10)

    plt.figure(figsize=(8, 7))
    plt.tight_layout()
    plt.plot(k, data["HH"], 'r-', label = 'HH')
    plt.plot(k, data["Neu"], 'b-', label = 'Neu')
    plt.plot(k, data["Ext"], 'g-', label = 'Ext')
    plt.plot(k, data["Agr"], 'c-', label = 'Agr')
    plt.plot(k, data["Con"], 'm-', label = 'Con')
    plt.plot(k, data["Ope"], 'k-', label = 'Ope')
    plt.xticks(k, x_stick, rotation=45)
    plt.legend(loc = 'lower right')
    plt.ylabel("users number")
    plt.xlabel("big-6 scores")
    plt.savefig("pr_dist.eps", format='eps', dpi=1000)
    plt.show()

if __name__ == '__main__':
    main()
