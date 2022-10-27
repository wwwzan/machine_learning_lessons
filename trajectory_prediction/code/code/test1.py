import os
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import math
from math import radians, cos
from util import query_wind, query_current
import time
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split  # 分出test

warnings.filterwarnings('ignore')


# 计算与xx分钟之前位置的一阶导
# 基于梯度的单边采样（GOSS）
def calculate_diff(data, diff_time, rolling_time, time_range):
    data.set_index('datetime', inplace=True)
    for i in range(time_range[0], time_range[1]):
        data['U10_{}'.format(i)] = data['U10_{}'.format(i)].rolling(rolling_time).mean()
        data['V10_{}'.format(i)] = data['V10_{}'.format(i)].rolling(rolling_time).mean()
        data['u_{}'.format(i)] = data['u_{}'.format(i)].rolling(rolling_time).mean()
        data['v_{}'.format(i)] = data['v_{}'.format(i)].rolling(rolling_time).mean()

    data['lng_diff'] = data['lng'].diff(diff_time).rolling(rolling_time).mean()
    data['lat_diff'] = data['lat'].diff(diff_time).rolling(rolling_time).mean()
    return data.iloc[diff_time:]


def in_area(lat, lng, in_fix_area, fix_area):
    if in_fix_area == 1:
        return in_fix_area
    lat1, lng1, lat2, lng2 = fix_area
    return 1 if lat < lat1 and lat > lat2 and lng > lng1 and lng < lng2 else 0


def first_point(data):
    return data.iloc[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="../prediction_result",
                        help="path of output file")  # 将结果保存在另一个文件夹
    parser.add_argument("--diff_time", type=int, default=30, help="diff time")
    parser.add_argument("--rolling_time", type=str, default="1min", help="rolling time")
    parser.add_argument("--interpolation", type=str, required=True, choices=['cubic', 'linear'],
                        help="use which interpolation method")
    parser.add_argument("--nearest_circle", type=int, default=3, help="interpolation range")
    parser.add_argument("--time_start", type=int, default=-4, help="start time")
    parser.add_argument("--time_end", type=int, default=4, help="end time")
    args = parser.parse_args()

    output_dir = args.output
    diff_time = args.diff_time
    rolling_time = args.rolling_time
    kind = args.interpolation
    nearest_circle = args.nearest_circle
    time_start = args.time_start
    time_end = args.time_end

    # 返回X小时前到X小时后此地的wind, current数据
    time_range = [time_start, time_end + 1]

    # ./当前目录 ../上级目录
    data_path = '../user_data'
    save_path = '../prediction_result/'
    current_path = '../raw_data/current'
    wind_path = '../raw_data/wind'
    # 岸边数据
    fix_areas = [
        [25.55, 118.38, 24.83, 118.70],
        [29.15, 121.84, 29.10, 121.90],
        [25.30, 121.58, 25.30, 121.60],
        [25.70, 119.65, 25.60, 119.80],
        [24.56, 118.36, 24.55, 118.40],
        [25.51, 119.31, 25.40, 119.45],
        [26.68, 119.91, 26.56, 120.05],
        [27.35, 120.22, 27.25, 120.35],
    ]

    # 训练数据
    print("train")
    data = pd.read_csv(os.path.join(data_path, 'train.csv'), parse_dates=['datetime'])
    data['in_fix_area'] = 0
    df = data.groupby('id').apply(first_point).reset_index(drop=True)
    # 去除在岸边的数据
    for fix_area in fix_areas:
        df.loc[df.index, 'in_fix_area'] = df.apply(lambda x: in_area(x.lat, x.lng, x.in_fix_area, fix_area), axis=1)
    data = data[data['id'].isin(df[df['in_fix_area'] == 0]['id'])]

    # 测试数据
    print("test")
    test = pd.read_csv('../raw_data/test.csv')
    test['datetime'] = test['date'] + '-' + test['time']
    test['datetime'] = test['datetime'].map(lambda x: datetime.strptime(x, '%Y-%m-%d-%H:%M'))
    test['in_fix_area'] = 0
    # 去除在岸边的数据
    for fix_area in fix_areas:
        test.loc[test.index, 'in_fix_area'] = test.apply(lambda x: in_area(x.lat, x.lng, x.in_fix_area, fix_area),
                                                         axis=1)
    fix_ids = test[test['in_fix_area'] == 1]['id'].to_list()

    data = data.groupby('id').apply(calculate_diff, diff_time=diff_time, rolling_time=rolling_time,
                                    time_range=time_range).reset_index(level='id', drop=True)

    # 构建lat和东西向速度的组合特征
    data['lat_code'] = data['lat'].map(radians).map(cos)
    for i in range(time_range[0], time_range[1]):
        data['U10_{}'.format(i)] = data['U10_{}'.format(i)] / data['lat_code']
        data['u_{}'.format(i)] = data['u_{}'.format(i)] / data['lat_code']

    # 只用当下时刻的wind和current数据
    features_lng = ['U10_0', 'u_0']

    target_lng = 'lng_diff'

    train = data.copy()

    # 训练经度模型
    x_train, x_test, y_train, y_test = train_test_split(train[features_lng], train[target_lng], test_size=0.2,
                                                        random_state=3)

    models = [LinearRegression(), KNeighborsRegressor(), SVR(), Ridge(), Lasso(), MLPRegressor(alpha=20),
              DecisionTreeRegressor(), ExtraTreeRegressor(), XGBRegressor(), RandomForestRegressor(),
              AdaBoostRegressor(), GradientBoostingRegressor(), BaggingRegressor(),]
    models_str = ['LinearRegression', 'KNNRegressor', 'SVR', 'Ridge', 'Lasso', 'MLPRegressor', 'DecisionTree',
                  'ExtraTree', 'XGBoost', 'RandomForest', 'AdaBoost', 'GradientBoost', 'lightBGM']
    score_ = []

for name, model in zip(models_str, models):
    print('开始训练模型：' + name)
    model = model  # 建立模型
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = model.score(x_test, y_test)
    score_.append(str(score)[:5])
    print(name + ' 得分:' + str(score))
