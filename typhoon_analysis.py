import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mpt
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import random
import numpy.ma as npm
import seaborn as sns


label = ['time', 'type', 'lat', 'lon', 'v', '24v', 'VWS', 'RH600', 'DIV200', 'VOR850', 'SST', 'MPI', 'TO', 'Forecast']
preindex = {"0": 23582, "12": 21720, "24": 19920, "36": 18152, "48": 16422, "60": 14760, "72": 13180, "84": 11678,
            "96": 10245, "108": 8915, "120": 7697, "132": 6598, "144": 5611}


class Typhoon(object):
    def __init__(self):
        self.data = []
        self.time = np.array([])
        self.lon = np.array([])
        self.lat = np.array([])
        self.ri = np.array([])
        self.prediction = np.array([])
        self.sample_hours = 6.0

    def feed_data(self, data_line):
        self.data.append(data_line)

    def is_mine(self, data_line):
        if len(self.data) > 0:
            latest = self.data[-1][0]
            sample_hours = (data_line[0] - latest).total_seconds() / 3600
            if sample_hours != self.sample_hours:
                # print("WARNING: sampling time is incorrect, there might be any data missing!")
                lat = data_line[2]
                lon = data_line[3]
                latest_lat = self.data[-1][2]
                latest_lon = self.data[-1][3]
                if abs(latest_lat - lat) > 4 or abs(latest_lon - lon) > 4:
                    return False
            return True
        else:
            return True

    def read_data(self):
        self.time = self.data[:, 0]
        self.lat = self.data[:, 2]
        self.lon = self.data[:, 3]
        self.ri = self.data[:, 5]
        self.prediction = self.data[:, 13]


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


def clas(hours, estimator=1000):
    data = np.loadtxt("./%d.txt" % hours, dtype=float)
    df = pd.DataFrame(data, columns=label)
    # df['time'] = factor[:, 0].astype('int32')
    df = df[0:preindex["%d" % hours]]  # 2016:23582
    df_ori = df_maker(df, True)

    train_features, test_features, train_labels, test_labels, feature_list = data_processor(df_ori, 0.25)
    rf = RandomForestClassifier(n_estimators=estimator, oob_score=True, class_weight="balanced")
    rf.fit(train_features, train_labels)

    df = pd.DataFrame(data, columns=label)
    # df['time'] = factor[:, 0].astype('int32')
    df = df[preindex["%d" % hours]:]  # 2016:23582
    df = df_maker(df)
    train_features, test_features, train_labels, test_labels, feature_list = data_processor(df, 0.99)
    coll = (rf, train_features, test_features, train_labels, test_labels, feature_list)
    return coll


def all_accuracy(predictions, test_labels):
    return metrics.accuracy_score(test_labels, predictions)


def pod(predictions, test_labels):
    mask = test_labels != 1
    test_labels_1 = npm.masked_array(test_labels, mask=mask)
    predictions_1 = npm.masked_array(predictions, mask=mask)
    result = (predictions_1 == test_labels_1)
    RI_accuracy = np.sum(result != 0) / (np.sum(result != 0) + np.sum(result == 0))
    RI_counts = np.sum(result != 0) + np.sum(result == 0)
    return RI_counts, RI_accuracy


def ori_accuracy(predictions, test_labels):
    mask = test_labels == 1
    test_labels_2 = npm.masked_array(test_labels, mask=mask)
    predictions_2 = npm.masked_array(predictions, mask=mask)
    result = (predictions_2 == test_labels_2)
    Ori_accuracy = np.sum(result != 0) / (np.sum(result != 0) + np.sum(result == 0))
    return Ori_accuracy


def far(predictions, test_labels):
    mask = predictions != 1
    test_labels_0 = npm.masked_array(test_labels, mask=mask)
    predictions_0 = npm.masked_array(predictions, mask=mask)
    result = predictions_0 != test_labels_0
    far_rate = np.sum(result != 0) / np.sum(test_labels == 0)
    return far_rate


def result_processor(rf, test_labels, test_features, feature_list):
    oob_score = 1 - rf.oob_score_
    # print("oob_error:", oob_score)
    predictions = rf.predict(test_features)
    print("All_accuracy:", all_accuracy(predictions, test_labels))
    RI_counts, RI_accuracy = pod(predictions, test_labels)
    print("There are %d RI" % RI_counts)
    print("RI_accuracy :", RI_accuracy)
    print("Ori_accuracy :", ori_accuracy(predictions, test_labels))
    print("FAR:", far(predictions, ))

    # importances = list(rf.feature_importances_)
    # # List of tuples with variable and importance
    # feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # # Sort the feature importances by most important first
    # feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # # Print out the feature and importances
    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # rf_fpr, rf_tpr, rf_thresholds = roc_curve(test_labels, rf.predict_proba(test_features)[:, 1])
    # plt.figure()
    # plt.plot(rf_fpr, rf_tpr, label='RF')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Graph')
    # plt.legend(loc="lower right")
    # plt.show()


def plot(typhoon):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 使地图成为全局地图，而不是将其范围放大到任何绘制数据的范围
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.LAND)
    ax.coastlines()
    # ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree(180))
    # ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree(180))
    extent = [100, 180, 0, 30]  # 经纬度范围
    ax.set_extent(extent)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.2,
        color='k',
        alpha=0.5,
        linestyle='--'
    )
    gl.top_labels = False  # 关闭顶端的经纬度标签
    gl.right_labels = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mpt.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mpt.FixedLocator(np.arange(-90, 91, 10))
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    plt.plot(typhoon.lon, typhoon.lat, markersize=12, transform=ccrs.PlateCarree())
    # plt.title('shortwave_all_surface')
    plt.show()


def sel_example(data, n=10):
    example = []
    sel = np.random.choice(np.array([i for i in range(len(data))]), n, replace=False)
    for i in range(n):
        example.append(data[sel[i]])
    example = np.array(example)
    return example


def example(pre, feature, label):
    RI_index = np.where(label == 1)
    RI_label = np.array(feature)[RI_index]
    pre_label = np.array(pre)[RI_index]
    right_index = np.where(pre_label == 1)
    error_index = np.where(pre_label != 1)
    right_label = RI_label[right_index]
    error_label = RI_label[error_index]
    right_example = sel_example(right_label)
    if len(error_label) > 10:
        err_example = sel_example(error_label)
    else:
        err_example = error_label

    pre_index = np.where(pre == 1)
    pre_l = pre[pre_index]
    l_l = np.array(label)[pre_index]
    fea_l = np.array(feature)[pre_index]
    result = pre_l != l_l
    wrong_index = np.where(result == True)
    fpr = fea_l[wrong_index]
    if len(pre_l) > 10:
        fpr = sel_example(fpr)
    else:
        pass

    right_example[:, 9] = - right_example[:, 9]
    err_example[:, 9] = - err_example[:, 9]
    fpr[:, 9] = -fpr[:, 9]
    return right_example, err_example, fpr


def draw_feature_scatter(right_example, err_example, n=10):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    feature_list.pop(-1)
    feature_list.append('-TO')
    delta = 0.05
    xaxis = np.array([i for i in range(10)])
    for i in range(n):
        plt.scatter(xaxis + delta, right_example[i], s=10, c='b')
        plt.scatter(xaxis - delta,  err_example[i], s=10, c='r')
    ax.set_xticks(xaxis)
    ax.set_xticklabels(feature_list)
    ax.legend(['True', 'False'])
    ax.set_title('Example Feature')
    plt.savefig("Example_Feature.png")
    plt.show()


def draw_plot(data, offset, edge_color, fill_color):
    pos = np.arange(data.shape[1])+offset
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
    return bp


def draw_ex():
    coll = clas(0)
    rf, train_features, test_features, train_labels, test_labels, feature_list = coll
    prediction = rf.predict(test_features)
    right_example, err_example, fdr_example = example(prediction, test_features, test_labels)
    feature_list.pop(-1)
    feature_list.append('-TO')
    fig, ax = plt.subplots()
    a = draw_plot(right_example, -0.3, "tomato", "white")
    b = draw_plot(err_example, +0.3, "skyblue", "white")
    c = draw_plot(fdr_example, 0, "green", "white")
    ax.set_xticks([i for i in range(10)])
    ax.set_xticklabels(feature_list)
    ax.legend([a["boxes"][0], b["boxes"][0], c["boxes"][0]], ["Accuracy", "FNR", "FPR"])
    plt.savefig("example.png")
    plt.show()


if __name__ == '__main__':
    coll = clas(0)
    rf, train_features, test_features, train_labels, test_labels, feature_list = coll
    prediction = rf.predict(test_features)
    ac_example, fnr_example, fpr_example = example(prediction, test_features, test_labels)
    feature_list.pop(-1)
    feature_list.append('-TO')
    fig, ax = plt.subplots()
    a = draw_plot(ac_example, -0.3, "tomato", "white")
    b = draw_plot(fnr_example, +0.3, "skyblue", "white")
    c = draw_plot(fpr_example, 0, "green", "white")
    ax.set_xticks([i for i in range(10)])
    ax.set_xticklabels(feature_list)
    ax.legend([a["boxes"][0], b["boxes"][0], c["boxes"][0]], ["Accuracy", "FNR", "FPR"])
    plt.show()
