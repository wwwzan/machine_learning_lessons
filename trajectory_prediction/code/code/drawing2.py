import warnings
import matplotlib.pyplot as plt
import pandas as pd
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.rcParams['axes.unicode_minus'] = False

def calculate_10000diff(data):
    # 因为1分钟内经纬度的变化很小, 为了便于与风速和流速一同观察, 这里将经纬度偏移量乘上了10000的系数
    data['lng_diff'] = data['lng'].diff() * 10000
    data['lat_diff'] = data['lat'].diff() * 10000
    return data.iloc[1:]

def show_relation(data, id):
    flow = data.groupby('id').apply(calculate_10000diff).reset_index(drop=True).set_index('datetime')
    plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(121)
    ax1.plot(flow[flow["id"] == id]['u_0'].asfreq('min').rolling('15min').mean(), label='洋流的东向分量')
    ax1.plot(flow[flow["id"]==id]['U10_0'].asfreq('min').rolling('15min').mean(), label='风的东向分量')
    ax1.plot(flow[flow["id"]==id]['lng_diff'].asfreq('min').rolling('15min').mean(), label='经度偏移量')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.ylim([-10, 10])
    ax1.legend()
    ax1.set_title('东西分布 ({}, {})'.format(id, data.groupby('id')['datetime'].agg(min)[id]),
                  fontweight='bold')
    ax2 = plt.subplot(122)
    ax2.plot(flow[flow["id"] == id]['v_0'].asfreq('min').rolling('15min').mean(), label='洋流的北向分量')
    ax2.plot(flow[flow["id"] == id]['V10_0'].asfreq('min').rolling('15min').mean(), label='风的北向分量')
    ax2.plot(flow[flow["id"] == id]['lat_diff'].asfreq('min').rolling('15min').mean(), label='纬度偏移量')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.ylim([-10, 10])
    ax2.legend()
    ax2.set_title('南北分布 ({}, {})'.format(id, data.groupby('id')['datetime'].agg(min)[id]),
                  fontweight='bold')
    plt.show()

if __name__ == '__main__':
    data_path = '../user_data'
    data = pd.read_csv(os.path.join(data_path, 'train.csv'), parse_dates=['datetime'])
    show_relation(data, id=582401)