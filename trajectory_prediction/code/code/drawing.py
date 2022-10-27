import folium
import os
import pandas as pd
import numpy as np


def read_gps_data(path):
    P = pd.read_csv(path).values  # 读取csv文件，输出为narray
    # narray转换成list
    # locations = P[0:387, [4,3]].tolist() #第一个
    # locations = P[387:855,[4,3]].tolist() #第二个
    # locations = P[855:2295, [4, 3]].tolist()
    # locations = P[2295:3267, [4, 3]].tolist()
    locations = P[:, [4, 3]].tolist()

    return locations


def draw_gps(locations, output_path, file_name):
    """
    绘制gps轨迹图
    :param locations: list, 需要绘制轨迹的经纬度信息，格式为[[lat1, lon1], [lat2, lon2], ...]
    :param output_path: str, 轨迹图保存路径
    :param file_name: str, 轨迹图保存文件名
    :return: None
    """
    m = folium.Map(locations[0], zoom_start=30, attr='default')  # 中心区域的确定

    folium.PolyLine(  # polyline方法为将坐标用实线形式连接起来
        locations,  # 将坐标点连接起来
        weight=2,  # 线的大小为4
        color='red',  # 线的颜色为红色
        opacity=0.6,  # 线的透明度
    ).add_to(m)  # 将这条线添加到刚才的区域m内


    # 起始点，结束点
    #folium.Marker(locations[0], popup='<b>Starting Point</b>').add_to(m)
    #folium.Marker(locations[-1], popup='<b>End Point</b>').add_to(m)
    m.add_child(folium.LatLngPopup())
    m.save(os.path.join(output_path, file_name))  # 将结果以HTML形式保存到指定路径
    # 解决在国内打不开网页的问题
    search_text = "cdn.jsdelivr.net"
    replace_text = "gcore.jsdelivr.net"


if __name__ == '__main__':
    path1 = '../raw_data/train.csv'  # 4,5列是经纬度

    locations = read_gps_data(path1)
    draw_gps(locations,  '../../img', 'guiji.html')

