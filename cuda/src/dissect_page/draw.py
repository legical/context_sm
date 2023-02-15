#!/usr/bin/env python3
from cProfile import label
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
# example: python3 draw.py 3060


def readname(filePath):
    # default: ./data-3060/
    # filePath = './data-'+GPU_name+'/'
    name = os.listdir(filePath)
    csv = []
    for file in name:
        if ".csv" in file:
            csv.append(file)
    return csv

# get inner_cycle from filename


def get_inner(filename):
    i = filename.find('inner')
    if i > 0:
        tail = filename.find('.', i, i+9)
        return int(filename[i+5:tail])
    return 1


def get_data(filename, EXEClist, hit_rate):
    # '''get the highs and lows from a data file'''
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        for row in reader:
            try:
                time = float(row[1])
                rate = float(row[5])
                # 去除 cache hit rate > 100 的数据
                if rate <= 100:
                    EXEClist.append(time)
                    hit_rate.append(rate)
            except ValueError:
                print(row[0], 'reading data error!\n')


# default : 3060
GPU_name = "3060"
if len(sys.argv) > 1:
    GPU_name = str(sys.argv[1])


# default: ./data-3060/
DATA_DIR = './data-'+GPU_name+'/'
PIC_DIR='./data-'+GPU_name+'/pic/'

csvlist = readname(DATA_DIR)
# 创建了一个空的7列的二维数组
# inner time_min time_max time_avg hit_rate_min hit_rate_max hit_rate_avg
data_analysis = np.empty(shape=[0, 7], dtype=float)
# x=np.append(x,[[1,2,3,4]],axis=0)#添加整行元素，axis=1添加整列元素

# print(csvlist)
for file in csvlist:
    # 内部循环次数
    inner = get_inner(file)
    filename = DATA_DIR+file
    # print("\n",filename)
    # 图片dpi=220，尺寸宽和高，单位为英寸
    fig = plt.figure(dpi=220, figsize=(80, 32))
    ax1 = fig.add_subplot(111)

    # skiprows=1 跳过标题行
    EXEClist, hit_rate = [], []
    get_data(filename, EXEClist, hit_rate)
    # 添加当前inner的数据
    data_analysis = np.append(data_analysis, [[inner, min(EXEClist), max(EXEClist), np.mean(
        EXEClist), min(hit_rate), max(hit_rate), np.mean(hit_rate)]], axis=0)
    IDlist = np.arange(1, len(EXEClist)+1)
    ax1.plot(IDlist, EXEClist, "g", marker='D',
             markersize=5, label="Execution time")

    # y轴刻度值
    # plt.yticks(np.arange(min_y_lable, max_y_lable, 0.2))
    # ax1.set_ylim((np.floor(minax[0]), np.ceil(minax[1])))
    ax1.set_xlabel('Index', fontsize=36)
    ax1.set_ylabel('EXEC_time', fontsize=36)
    # 控制图例的形状大小：fontsize控制图例字体大小，markerscale控制scatters形状大小，scatterpoints控制scatters的数量
    ax1.legend(loc=2, fontsize=32, scatterpoints=1)
    # 设置 y 轴显示网格线
    ax1.grid(axis='y')

    ax2 = ax1.twinx()  # 创建第二个坐标轴，L2 Hit Rate
    ax2.plot(IDlist, hit_rate, 'o-', c='blue',
             markersize=5, label="L2 Hit Rate%", linewidth=0.4)
    ax2.set_ylabel('L2 Hit Rate', fontsize=36)
    # ax2.set_yticks([GPU_addr_list[0], GPU_addr_list[-1]])
    # 控制图例的形状大小：fontsize控制图例字体大小，markerscale控制scatters形状大小，scatterpoints控制scatters的数量
    ax2.legend(loc=1, fontsize=32, scatterpoints=1)
    ax2.grid(visible=False)

    plt.tick_params(labelsize=32)  # 刻度字体大小
    # plt.title('执行时间折线图')  # 折线图标题
    chart_title = 'inner={} * 4KB = Data Size   ^v^    Time||min={}   max={}   avg={}    ###  L2 Hit Rate%||min={}   max={}   avg={}'
    plt.title(chart_title.format(
        inner, min(EXEClist), max(EXEClist), np.mean(EXEClist), min(hit_rate), max(hit_rate), np.mean(hit_rate)), fontsize=46)
    # plt.gcf().autofmt_xdate()
    filename = filename.replace(DATA_DIR, PIC_DIR, 1)
    pic_name = filename.replace("csv", "jpg", 1)
    # 如果图片文件已存在，则删除
    if os.path.exists(pic_name):
        os.remove(pic_name)
    plt.savefig(pic_name)
    # plt.show()

# 创建两个子图 -- 图3
f, (time, hit) = plt.subplots(2, 1, figsize=(40, 34))
data_analysis = np.array(data_analysis)
inner_list = data_analysis[:, 0]
time_min = data_analysis[:, 1]
time_max = data_analysis[:, 2]
time_avg = data_analysis[:, 3]
hit_min = data_analysis[:, 4]
hit_max = data_analysis[:, 5]
hit_avg = data_analysis[:, 6]

time.plot(inner_list, time_min, "g", marker='^',
          markersize=5, label="time_min")
time.plot(inner_list, time_max, "r", marker='v',
          markersize=5, label="time_max")
time.plot(inner_list, time_avg, "y", marker='o',
          markersize=5, label="time_avg")

# x,y轴标签
time.set_xlabel('inner * 4KB = Data Array Size', fontsize=36)
time.set_ylabel('EXEC_time', fontsize=36)
# 控制图例的形状大小：fontsize控制图例字体大小，markerscale控制scatters形状大小，scatterpoints控制scatters的数量
time.legend(loc=2, fontsize=32, scatterpoints=1)
# 设置 y 轴显示网格线
time.grid(axis='y')
plt.tick_params(labelsize=32)  # 刻度字体大小

hit.plot(inner_list, hit_min, "g", marker='^',
         markersize=5, label="hit_rate_min")
hit.plot(inner_list, hit_max, "r", marker='v',
         markersize=5, label="hit_rate_max")
hit.plot(inner_list, hit_avg, "y", marker='o',
         markersize=5, label="hit_rate_avg")

# x,y轴标签
hit.set_xlabel('inner * 4KB = Data Array Size', fontsize=36)
hit.set_ylabel('L2 Hit Rate%', fontsize=36)
# 控制图例的形状大小：fontsize控制图例字体大小，markerscale控制scatters形状大小，scatterpoints控制scatters的数量
hit.legend(loc=2, fontsize=32, scatterpoints=1)
# 设置 y 轴显示网格线
hit.grid(axis='y')

plt.tick_params(labelsize=32)  # 刻度字体大小
# 折线图标题
plt.title('Data Analysis')
# plt.gcf().autofmt_xdate()
pic_name = PIC_DIR + "data_analysis.jpg"
# 如果图片文件已存在，则删除
if os.path.exists(pic_name):
    os.remove(pic_name)
plt.savefig(pic_name)
