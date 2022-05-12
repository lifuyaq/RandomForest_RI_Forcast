import numpy as np
import numpy.ma as npm
import pandas as pd
import scipy.io as sio
from matplotlib import ticker
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import normalize

label = ['time', 'type', 'lat', 'lon', 'v', '24v', 'VWS', 'RH600', 'DIV200', 'VOR850', 'SST', 'MPI', 'TO']

WNP24 = sio.loadmat('data/factor.mat')
factor = WNP24['factor']
df = pd.DataFrame(factor, columns=label)
df['time'] = factor[:, 0].astype('int32')
df['24v'] = np.where(df['24v'] >= 15, 1, 0)
RI_index = np.where(df['24v'] == 1)
NONRI_index = np.where(df['24v'] != 1)
df = np.array(df)
RI_data = df[RI_index]
NONRI_data = df[NONRI_index]


def salience(ri, nonri):
    statistics, pvalue = stats.ttest_ind(ri, nonri, axis=0, equal_var=False)
    with open("factor_analysis", "w") as f:
        for i in range(len(label)):
            RI_mean = RI_data[:, i].mean()
            NON_mean = NONRI_data[:, i].mean()
            print("%s_mean:RI %f" % (label[i], RI_mean), file=f)
            print("%s_mean:NONRI %f" % (label[i], NON_mean), file=f)
            print("%s_meandiff:RI %f" % (label[i], NON_mean - RI_mean), file=f)
            print("%s_pvalue: %f" % (label[i], pvalue[i]), file=f)
            print("", file=f)


def draw_hist(ri, nonri, title):
    fig, ax = plt.subplots()
    ax.grid(axis='y')
    ax.set_title("%s" % title, y=-0.15)
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    plt.hist([ri, nonri], label=['RI', 'nonRI'], density=True, color=['black', 'w'], edgecolor='black')
    plt.legend(loc="upper right")
    plt.savefig("result/%s.png" % title)
    plt.show()
    plt.close()


def BDI(ri, nonri):
    bdi = (ri.mean()-nonri.mean())/(np.std(ri)+np.std(nonri))
    return bdi


def draw_bdi(ri, nonri):
    fig, ax = plt.subplots()
    ax.grid(axis='y')
    ax.set_title("BDI_score")
    ax.set_ylabel("BDI_score")
    ax.set_xticklabels([label[4]]+label[6:])
    bdi_score = [BDI(ri[:, i], nonri[:, i]) for i in range(4, 13)]
    bdi_score.pop(1)
    plt.bar([label[4]]+label[6:], bdi_score, color='black')
    plt.savefig("BDI_score.png")
    plt.show()



if __name__ == '__main__':
    draw_bdi(RI_data, NONRI_data)



