import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = []
    s_ratio = pd.read_csv("Shopping_ratio.csv")
    t_ratio = pd.read_csv('Travel_ratio.csv')
    day_len = np.array(len(s_ratio))
    day = np.arange(day_len)
    x = [1, 32, 62, 93, 123]
    y = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
    y_label = ['0.00', '0.02', '0.04', '0.06', '0.08', '0.10', '0.12', '0.14']
    x_label = ["Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize=(8,4))
    sc_x1 = [145]
    sc_x2 = [146]
    sc_y1 = [0.082]
    sc_y2 = [0.0058]
    plt.plot(day, s_ratio['ratio'], 'C1', label = 'shopping ratio')
    plt.plot(day, t_ratio['ratio'], 'C2', label = 'travel ratio')
    plt.scatter(sc_x1, sc_y1, c='red', label = 'Christmas')
    plt.scatter(sc_x2, sc_y2, c='red')
    plt.ylabel('ratio')
    plt.legend(loc = 'upper right')
    plt.xticks(x, x_label)
    plt.yticks(y, y_label)
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    plt.savefig("shopping_travel_vibrate.pdf", format='pdf', dpi=1000)
    plt.show()




if __name__ == '__main__':
    main()
