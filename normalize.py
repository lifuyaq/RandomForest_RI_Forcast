import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


label = ['time', 'type', 'lat', 'lon', 'v', '24v', 'VWS', 'RH600', 'DIV200', 'VOR850', 'SST', 'MPI', 'TO', 'Forecast']


if __name__ == '__main__':
    hours = 0
    data = np.loadtxt("./%d.txt" % hours, dtype=float)
    df = pd.DataFrame(data, columns=label)
    # df = normalize(df, axis=0, norm='max')
    df = np.array(df)
    max = np.max(df, axis=0)
    min = np.min(df, axis=0)
    judge = np.array([0, 0, 0, 0, 0, 0, 7, 65, 5, 10, 28, 70, -60, 0])
    j_value = (judge - min) / (max - min)
    with open("j_value.txt", "w") as f:
        for i in range(len(label)):
            if 5 < i < 13:
                print("%s:%f" % (label[i], j_value[i]), file=f)

