import random
import scipy.io as sio
import numpy as np
import numpy.ma as npm
import pandas as pd
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.metrics import roc_curve

label = ['time', 'type', 'lat', 'lon', 'v', '24v', 'VWS', 'RH600', 'DIV200', 'VOR850', 'SST', 'MPI', 'TO']


def time_transformer(time):  # time_type:int 1980040500
    year = time // 1000000
    month = (time - year * 1000000) // 10000
    day = (time - year * 1000000 - month * 10000) // 100
    hour = time % 100
    return datetime.datetime(year, month, day, hour)


def time_calculate(time1, time2):  # time_type:int 1980040500
    return time_transformer(time1) - time_transformer(time2)


def seperate(original_data):  # ta_type = CMA
    new_data = [[]]
    example_counts = 0
    for i in range(original_data.shape[0]):
        if i == 0:
            new_data[example_counts].append(original_data[0])
            continue
        time_delta = time_calculate(original_data[i][0], original_data[i - 1][0])
        if time_delta.seconds == 21600 and time_delta.days == 0 or original_data[i][1] > 1:
            new_data[example_counts].append(original_data[i])
        else:
            example_counts += 1
            new_data.append([])
            new_data[example_counts].append(original_data[i])
    return new_data


def time_list(seperated_data, type="normal"):
    time_list = []
    if type == "normal":
        for i in range(len(seperated_data)):
            time_list.append(time_transformer(seperated_data[i][0][0]))
    else:
        for i in range(len(seperated_data)):
            for k in range(len(seperated_data[i])):
                if seperated_data[i][k][5] >= 15:
                    time_list.append(time_transformer(seperated_data[i][k][0]))
    return time_list


def time_distribution(time_list):
    year_counts = []
    years = []
    month = [0] * 12
    delta_year = 0
    for i in range(len(time_list)):
        if i == 0:
            year_counts.append(1)
            years.append(time_list[i].year)
        elif time_list[i].year == time_list[i - 1].year:
            year_counts[delta_year] = year_counts[delta_year] + 1
        else:
            year_counts.append(1)
            years.append(time_list[i].year)
            delta_year += 1
        month[time_list[i].month - 1] += 1
    return years, year_counts, month


def time_plot(data, type="month"):
    if type == "month":
        MONTH = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        fig, ax = plt.subplots()
        ax.bar(MONTH, data, color='k')
        ax.set_xlabel("Month")
        ax.set_ylabel("Frequency")
        ax.set_title("Monthly Distribution")
        fig.savefig("Monthly_distribution.png")
        plt.show()
    else:
        fig, ax = plt.subplots()
        ax.bar(data[0], data[1], color='k')
        ax.set_xlabel("Year")
        ax.set_ylabel("Frequency")
        ax.set_title("Annual Distribution")
        fig.savefig("Annual_distribution")
        plt.show()


def WNP_percentile(data):
    """data:original_data"""
    percentile = np.percentile(data[..., 5], np.linspace(0, 100, 21))
    return percentile


def percentile_draw(data):
    """data:original_data"""
    ALL_percentile = WNP_percentile(data)
    TD_percentile = WNP_percentile(data[data[..., 1] == 2])
    TS_percentile = WNP_percentile(data[data[..., 1] == 3])
    TY_percentile = WNP_percentile(data[data[..., 1] == 4])
    x = np.linspace(0, 100, 21)
    # x_smooth = np.linspace(0, 100, 21)
    # ALL_percentile_smooth = make_interp_spline(x, ALL_percentile)(x_smooth)
    # TD_percentile_smooth = make_interp_spline(x, TD_percentile)(x_smooth)
    # TS_percentile_smooth = make_interp_spline(x, TS_percentile)(x_smooth)
    # TY_percentile_smooth = make_interp_spline(x, TY_percentile)(x_smooth)
    fig, ax = plt.subplots()
    ax.plot(ALL_percentile, x, 'k-', TD_percentile, x, 'k:', TS_percentile, x, 'k-.', TY_percentile, x, 'k--')
    ax.legend(["All tropical cyclones", "Tropical depression", "Tropical storm", "Typhoon"])
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.linspace(-40, 40, 17))
    ax.set_yticks(np.linspace(0, 100, 11))
    ax.grid()
    ax.set_xlabel(r"$\Delta V_{24}(ms^{-1})$")
    ax.set_ylabel("Cumulative Frequency(%)")
    fig.savefig("Cumulative_Frequency.png")
    plt.show()


def location_get(data, fuzzy=False, extent=None):
    """ data:original_data
        extent = [lon_start, lon_stop, lat_start, lat_stop, lon_step, lat_step]
    """
    RI_location = data[data[..., 5] >= 15][..., 2:4]
    if fuzzy:
        flon = np.arange(extent[0], extent[1], extent[4])
        flat = np.arange(extent[2], extent[3], extent[5])
        new_data = np.zeros((len(flon), len(flat), 3))
        for i in range(len(flon)):
            new_data[i, ..., 0] = flon[i]
        for i in range(len(flat)):
            new_data[..., i, 1] = flat[i]
        for i in range(len(data)):
            lon_index = find_nearest(flon, data[i][3])
            lat_index = find_nearest(flat, data[i][2])
            new_data[lon_index][lat_index][2] += 1
        RI_location = new_data
    return RI_location


def location_draw(RI_location):
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(4, 4), dpi=550)
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
    # ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE, lw=0.3)
    extent = [90, 180, 0, 40]
    ax.set_extent(extent)
    ax.set_title("Distribution of RI", fontsize=6)
    lonlat_ticks(ax, extent, 10, 10, False)
    ax.tick_params(labelsize=5, length=1)
    # ax.scatter(RI_location[..., 0].flatten(), RI_location[..., 1].flatten(), s=RI_location[..., 2].flatten()/10, marker='.', c='k')
    ah = ax.hexbin(RI_location[..., 1], RI_location[..., 0], gridsize=15, extent=extent, cmap='pink_r', vmin=0, vmax=80)
    # ax.scatter(RI_location[..., 1], RI_location[..., 0], c='k', s=0.1, marker='.')
    cb = fig.colorbar(ah, ax=ax, pad=0.01, fraction=0.02)
    cb.ax.tick_params(labelsize=4, length=1)
    fig.savefig("location_distribution.png")
    plt.show()


def lonlat_ticks(ax, extent, dlon, dlat, gridlines=False):
    lon = extent[:2]
    lat = extent[2:]
    # dlon = (extent[1] - extent[0]) / gap
    # dlat = (extent[3] - extent[2]) / gap
    xticks = np.arange(lon[0], lon[1] + 0.1, dlon)
    yticks = np.arange(lat[0], lat[1] + 0.1, dlat)
    if gridlines:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, linestyle=':', color='k', alpha=0.8)
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())


def find_nearest(target_array, insert_value):
    return np.argmin(np.abs(target_array - insert_value))


def dup_rows(a, indx, num_dups=1):
    return np.insert(a, [indx + 1] * num_dups, a[indx], axis=0)


def dup_columns(a, indx, num_dups=1):
    return np.insert(a, [indx + 1] * num_dups, a[indx], axis=1)


def equal_data(df):
    RI_index = np.where(df['24v'] == 1)
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
    df['24v'] = np.where(df['24v'] >= 15, 1, 0)
    # df = df[0:23582]            #2016:23582

    df = df.drop('time', axis=1)
    df = df.drop('type', axis=1)

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

    labels = df['24v']
    df = df.drop('24v', axis=1)
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
    print("oob_error:", oob_score)
    predictions = rf.predict(test_features)
    # print("Accuracy", metrics.accuracy_score(test_labels, predictions))
    result = (predictions == test_labels)
    All_accuracy = np.sum(result != 0) / (np.sum(result != 0) + np.sum(result == 0))
    print("All_accuracy :", All_accuracy)
    mask = test_labels != 1
    test_labels_1 = npm.masked_array(test_labels, mask=mask)
    predictions_1 = npm.masked_array(predictions, mask=mask)
    result = (predictions_1 == test_labels_1)
    RI_accuracy = np.sum(result != 0) / (np.sum(result != 0) + np.sum(result == 0))
    print("RI_accuracy :", RI_accuracy)
    mask = test_labels == 1
    test_labels_2 = npm.masked_array(test_labels, mask=mask)
    predictions_2 = npm.masked_array(predictions, mask=mask)
    result = (predictions_2 == test_labels_2)
    Ori_accuracy = np.sum(result != 0) / (np.sum(result != 0) + np.sum(result == 0))
    print("Ori_accuracy :", Ori_accuracy)

    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

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

    return oob_score, metrics.accuracy_score(test_labels, predictions), All_accuracy, RI_accuracy, Ori_accuracy


def rf_predictor_regress():
    train_features, test_features, train_labels, test_labels, feature_list = data_processor()

    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)
    predictions = npm.array(predictions)
    test_labels = npm.array(test_labels)
    predictions = np.where(predictions > 0.15, 1, 0)
    mask = test_labels != 1
    predictions = npm.array(predictions, mask=mask)
    test_labels = npm.array(test_labels, mask=mask)
    result = (predictions == test_labels)
    print("All_accuracy :", np.sum(result != 0) / (np.sum(result != 0) + np.sum(result == 0)))

    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


def draw_score(data, data_all, labelname):
    fig, ax = plt.subplots()
    ax.set_title(labelname)
    plt.plot(label[2:5] + label[6:13], data)
    plt.plot(label[2:5] + label[6:13], [data_all] * 10)
    ax.legend(["without", "all"])

    plt.savefig("result/%s.png" % labelname)
    plt.show()


def rf_predictor_Classifier_factor_select():
    WNP24 = sio.loadmat('data/factor.mat')
    factor = WNP24['factor']
    df = pd.DataFrame(factor, columns=label)
    df['time'] = factor[:, 0].astype('int32')
    df_ori = df_maker(df, True)
    oob_score_list = []
    accuracy_score_list = []
    All_accuracy_list = []
    RI_accuracy_list = []
    Ori_accuracy_list = []

    for i in range(2, len(label)):
        if i == 5:
            train_features, test_features, train_labels, test_labels, feature_list = data_processor(df_ori, 0.25)
            rf = RandomForestClassifier(n_estimators=1000, oob_score=True, class_weight="balanced")
            rf.fit(train_features, train_labels)

            df = pd.DataFrame(factor, columns=label)
            df['time'] = factor[:, 0].astype('int32')
            df = df[23582:]  # 2016:23582
            df = df_maker(df)
            train_features, test_features, train_labels, test_labels, feature_list = data_processor(df, 0.99)

            oob_score_all, accuracy_score_all, All_accuracy_all, RI_accuracy_all, Ori_accuracy_all = result_processor(
                rf, test_labels,
                test_features,
                feature_list)
            all_data = [oob_score_all, accuracy_score_all, All_accuracy_all, RI_accuracy_all, Ori_accuracy_all]
        else:
            train_features, test_features, train_labels, test_labels, feature_list = data_processor(df_ori, 0.25,
                                                                                                    label[i])

            rf = RandomForestClassifier(n_estimators=1000, oob_score=True, class_weight="balanced")
            rf.fit(train_features, train_labels)

            df = pd.DataFrame(factor, columns=label)
            df['time'] = factor[:, 0].astype('int32')
            df = df[23582:]  # 2016:23582
            df = df_maker(df)
            train_features, test_features, train_labels, test_labels, feature_list = data_processor(df, 0.99, label[i])

            oob_score, accuracy_score, All_accuracy, RI_accuracy, Ori_accuracy = result_processor(rf, test_labels,
                                                                                                  test_features,
                                                                                                  feature_list)
            oob_score_list.append(oob_score)
            accuracy_score_list.append(accuracy_score)
            All_accuracy_list.append(All_accuracy)
            RI_accuracy_list.append(RI_accuracy)
            Ori_accuracy_list.append(Ori_accuracy)

    total = [oob_score_list, accuracy_score_list, All_accuracy_list, RI_accuracy_list, Ori_accuracy_list]
    total_name = ["oob_score", "accuracy_score", "All_accuracy", "RI_accuracy", "Ori_accuracy"]

    for k in range(len(total)):
        draw_score(total[k], all_data[k], total_name[k])


def rf_predictor_Classifier():
    WNP24 = sio.loadmat('data/factor.mat')
    factor = WNP24['factor']
    df = pd.DataFrame(factor, columns=label)
    df['time'] = factor[:, 0].astype('int32')
    df_ori = df_maker(df, True)

    train_features, test_features, train_labels, test_labels, feature_list = data_processor(df_ori, 0.25)
    rf = RandomForestClassifier(n_estimators=1000, oob_score=True, class_weight="balanced")
    rf.fit(train_features, train_labels)

    df = pd.DataFrame(factor, columns=label)
    df['time'] = factor[:, 0].astype('int32')
    df = df[23582:]  # 2016:23582
    df = df_maker(df)
    train_features, test_features, train_labels, test_labels, feature_list = data_processor(df, 0.99)
    result_processor(rf, test_labels, test_features, feature_list)


if __name__ == '__main__':

    data = WNP24['WNP24']
    data = seperate(data)
    time_list = time_list(data, "select")
    years, year_counts, month = time_distribution(time_list)
    time_plot([years, year_counts], 'year')
    # percentile_draw(data)
    # RI_location = location_get(data)
    # location_draw(RI_location)

    # rf_predictor_Classifier()
