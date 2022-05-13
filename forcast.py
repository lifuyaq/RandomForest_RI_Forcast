import random
import numpy as np
import numpy.ma as npm
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.metrics import roc_curve
from matplotlib import ticker

label = ['time', 'type', 'lat', 'lon', 'v', '24v', 'VWS', 'RH600', 'DIV200', 'VOR850', 'SST', 'MPI', 'TO', 'Forecast']
preindex = {"0": 23582, "12": 21720, "24": 19920, "36": 18152, "48": 16422, "60": 14760, "72": 13180, "84": 11678,
            "96": 10245, "108": 8915, "120": 7697, "132": 6598, "144": 5611}
unit = [r"$ m\cdot s^{-1} $", r"$ \% $", r"$ s^{-1} $", r"$ s^{-1} $", r"$ ^{\circ}$ C", r"$ m\cdot s^{-1} $", r"$ ^{\circ}$ C"]


def equal_data(df):
    RI_index = np.where(df['Forecast'] == 1)
    feature_list = list(df.columns)

    df = np.array(df)
    a = np.array(df)
    a = np.delete(a, RI_index, axis=0)
    random_index = random.sample(list(range(len(a))), np.size(RI_index))
    new_df = np.resize(np.array([0]), (0, len(feature_list)))
    new_df = np.append(new_df, df[RI_index], axis=0)
    for i in range(np.size(RI_index)):
        new_df = np.append(new_df, np.reshape(a[random_index[i]], (1, len(feature_list))), axis=0)

    df = pd.DataFrame(new_df, columns=feature_list)

    return df


def df_maker(df, data_select=False):
    df['Forecast'] = np.where(df['Forecast'] >= 15, 1, 0)

    df = df.drop('time', axis=1)
    df = df.drop('type', axis=1)
    df = df.drop('24v', axis=1)

    if data_select:
        df = equal_data(df)

    return df


def data_processor(df, test_size, drop_item=None):
    # df = df.drop('lat', axis=1)
    # df = df.drop('lon', axis=1)
    # df = df.drop('v', axis=1)
    # df = df.drop('VWS', axis=1)
    # df = df.drop('SST', axis=1)
    # df = df.drop('MPI', axis=1)
    # df = df.drop('VOR850', axis=1)
    # df = df.drop('RH600', axis=1)
    # df = df.drop('DIV200', axis=1)
    # df = df.drop('TO', axis=1)

    if drop_item:
        df = df.drop(drop_item, axis=1)

    labels = df['Forecast']
    df = df.drop('Forecast', axis=1)
    feature_list = list(df.columns)

    df = normalize(df, axis=0, norm='max')

    train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size=test_size)
    # print('Training Features Shape:', train_features.shape)
    # print('Training Labels Shape:', train_labels.shape)
    # print('Testing Features Shape:', test_features.shape)
    # print('Testing Labels Shape:', test_labels.shape)

    return train_features, test_features, train_labels, test_labels, feature_list


def result_processor(rf, test_labels, test_features, feature_list):
    oob_score = 1 - rf.oob_score_
    # print("oob_error:", oob_score)
    predictions = rf.predict(test_features)
    # print("Accuracy", metrics.accuracy_score(test_labels, predictions))
    # result = (predictions == test_labels)
    # All_accuracy = np.sum(result != 0) / (np.sum(result != 0) + np.sum(result == 0))
    # print("All_accuracy :", All_accuracy)

    mask = predictions != 1
    test_labels_0 = npm.masked_array(test_labels, mask=mask)
    predictions_0 = npm.masked_array(predictions, mask=mask)
    result = predictions_0 != test_labels_0
    far_rate = np.sum(result != 0) / np.sum(test_labels == 0)

    mask = test_labels != 1
    test_labels_1 = npm.masked_array(test_labels, mask=mask)
    predictions_1 = npm.masked_array(predictions, mask=mask)
    result = (predictions_1 == test_labels_1)
    RI_accuracy = np.sum(result != 0) / (np.sum(result != 0) + np.sum(result == 0))
    print("There are %d RI" % (np.sum(result != 0) + np.sum(result == 0)))
    print("RI_accuracy :", RI_accuracy)
    mask = test_labels == 1
    test_labels_2 = npm.masked_array(test_labels, mask=mask)
    predictions_2 = npm.masked_array(predictions, mask=mask)
    result = (predictions_2 == test_labels_2)
    Ori_accuracy = np.sum(result != 0) / (np.sum(result != 0) + np.sum(result == 0))
    # print("Ori_accuracy :", Ori_accuracy)

    # importances = list(rf.feature_importances_)
    # # List of tuples with variable and importance
    # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # # Sort the feature importances by most important first
    # feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # # Print out the feature and importances
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    rf_fpr, rf_tpr, rf_thresholds = roc_curve(test_labels, rf.predict_proba(test_features)[:, 1])
    # plt.figure()
    # plt.plot(rf_fpr, rf_tpr, label='RF')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Graph')
    # plt.legend(loc="lower right")
    # plt.show()

    return oob_score, metrics.accuracy_score(test_labels, predictions), far_rate, RI_accuracy, Ori_accuracy, rf_fpr, \
           rf_tpr, rf_thresholds


def classifier(hours, estimator=1000):
    data = np.loadtxt("./%d.txt" % hours, dtype=float)
    df = pd.DataFrame(data, columns=label)
    # df['time'] = factor[:, 0].astype('int32')
    df = df[0:preindex["%d" % hours]]            #2016:23582
    df_ori = df_maker(df, True)

    train_features, test_features, train_labels, test_labels, feature_list = data_processor(df_ori, 0.25)
    rf = RandomForestClassifier(n_estimators=estimator, oob_score=True, class_weight="balanced")
    rf.fit(train_features, train_labels)

    df = pd.DataFrame(data, columns=label)
    # df['time'] = factor[:, 0].astype('int32')
    df = df[preindex["%d" % hours]:]  # 2016:23582
    df = df_maker(df)
    train_features, test_features, train_labels, test_labels, feature_list = data_processor(df, 0.99)
    return result_processor(rf, test_labels, test_features, feature_list)


def average_accuracy(interval_time=24):
    # oob_score_list = []
    # accuracy_list = []
    leng = len([i for i in range(0, 108, interval_time)])
    All_accuracy_list = [np.array([])] * leng
    RI_accuracy_list = [np.array([])] * leng
    # Ori_accuracy_list = [np.array([])] * leng
    # rf_fpr_list = [np.array([])] * 5
    # rf_tpr_list = [np.array([])] * 5
    # rf_thr_list = [np.array([])] * 5
    for hours in range(0, 108, interval_time):
        rank = int(hours / interval_time)
        for i in range(100):
            oob_score, accuracy_score, All_accuracy, RI_accuracy, Ori_accuracy, rf_fpr, rf_tpr, rf_thresholds = classifier(hours)
            print("This is %d" % hours)
            # oob_score_list.append(oob_score)
            # accuracy_list.append(accuracy_score)
            All_accuracy_list[rank] = np.append(All_accuracy_list[rank], All_accuracy)
            RI_accuracy_list[rank] = np.append(RI_accuracy_list[rank], RI_accuracy)
            # Ori_accuracy_list[rank] = np.append(Ori_accuracy_list[rank], Ori_accuracy)


    # plt.figure()
    # for i in range(len(rf_fpr_list)):
    #     plt.plot(rf_fpr_list[i], rf_tpr_list[i], label='RF%d' % (i * 12))
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Graph')
    # plt.legend(loc="lower right")
    # plt.show()

    fig, ax = plt.subplots()
    plt.plot([hours for hours in range(0, 108, interval_time)], [RI_accuracy_list[average].mean() for average in range(leng)])
    plt.plot([hours for hours in range(0, 108, interval_time)], [All_accuracy_list[average].mean() for average in range(leng)])
    ax.legend(["POD", "FAR"])
    ax.set_title("Accuracy")
    ax.set_xlabel("Forecast_time")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xticks([i for i in range(0, 108, interval_time)])
    plt.savefig("Accuracy%d.png" % interval_time)
    # plt.show()


def oob_score():
        oob_score_list = [np.array([])] * 7
        for n in range(100, 2100, 300):
            rank = int((n - 100) / 300)
            for i in range(100):
                oob_score, accuracy_score, All_accuracy, RI_accuracy, Ori_accuracy, rf_fpr, rf_tpr, rf_thresholds = classifier(
                    0, estimator=n)
                oob_score_list[rank] = np.append(oob_score_list[rank], oob_score)

        fig, ax = plt.subplots()
        plt.plot([n for n in range(100, 2100, 300)], [average.mean() for average in oob_score_list])
        ax.set_title('oob_score')
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("oob_score")
        plt.savefig("oob_score2.png")
        plt.show()


def relatives(df):
    for i in range(0, 7):
        fig, ax = plt.subplots()
        ax.scatter(df[label[i + 6]], df[label[5]], s=0.05, c="black")
        x, y = percent(df[label[i + 6]], df[label[5]], 50)
        ax.plot(x, y, c="b")
        x, y = percent(df[label[i + 6]], df[label[5]], 95)
        ax.plot(x, y, c="r")
        ax.legend(["50th", "95th"])
        ax.set_xlabel("%s(%s)" % (label[i + 6], unit[i]))
        ax.set_ylabel(r"$ \Delta V24(m\cdot s^{-1}) $")
        # if not i:
            # plt.show()
        plt.savefig("%s_rela.png" % label[i + 6])
        # plt.show()


def percent_1(data, v, tile):
    min = data.min()
    max = data.max()
    x = np.linspace(min, max, 8) + (max-min)/16
    y = []
    pair = np.array([data, v])
    for i in range((len(x) - 1)):
        x1 = x[i]
        x2 = x[i+1]
        a = pair[:, np.where(pair[0] >= x1)][:, 0, :]
        b = pair[:, np.where(a[0] < x2)][:, 0, :]
        y.append(np.percentile(b[1], tile))

    return x[:-1], y


def percent(data, v, tile):
    min = data.min()
    max = data.max()
    indx = np.argsort(data)
    leng = len(data)
    x = data[indx]
    y = v[indx]
    plot_x = []
    plot_y = []
    interval = 1460 * 2
    for i in range(0, leng, interval):
        plot_x.append(x[i:(i+interval)].mean())
        plot_y.append(np.percentile(y[i:(i+interval)], tile))

    return plot_x, plot_y


if __name__ == '__main__':
    average_accuracy()
    average_accuracy(12)
