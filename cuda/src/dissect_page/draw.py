#!/usr/bin/env python3
from cProfile import label
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import csv
from matplotlib.ticker import ScalarFormatter
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
        tail = filename.find('.', i, len(filename))
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
                if int(sys.argv[1]) == 3060:
                    hit_num = float(row[5])
                    miss_num = float(row[6])
                    rate = 100*hit_num/(hit_num+miss_num)
                else:
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
PIC_DIR = './data-'+GPU_name+'/pic/'

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
    ax1.set_xlabel('Data of GPU : '+GPU_name, fontsize=42)
    ax1.set_ylabel('EXEC_time', fontsize=36)
    # 设置子图1的y轴刻度值用科学计数法表示
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.yaxis.offsetText.set_fontsize(28)
    # 设置指数为4
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
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

    plt.tick_params(labelsize=34)  # 刻度字体大小
    plt.setp(ax1.get_xticklabels(), fontsize=34)
    plt.setp(ax1.get_yticklabels(), fontsize=34)
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

# 创建3个子图 -- data_analysis
# axs[0]:time axs[1]:hit_rate axs[2]:origin data
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(80, 58))
# f, (time, hit) = plt.subplots(2, 1, figsize=(40, 34), sharex=True)
data_analysis = np.array(data_analysis)
# 所有数据的对比图，按照inner排序后再绘制
data_analysis = data_analysis[data_analysis[:, 0].argsort()]
inner_list = data_analysis[:, 0]
time_min = data_analysis[:, 1]
time_max = data_analysis[:, 2]
time_avg = data_analysis[:, 3]
hit_min = data_analysis[:, 4]
hit_max = data_analysis[:, 5]
hit_avg = data_analysis[:, 6]

axs[0].plot(inner_list, time_min, "g", marker='^',
            linewidth=3, markersize=15, label="time_min")
axs[0].plot(inner_list, time_max, "r", marker='v',
            linewidth=3, markersize=15, label="time_max")
axs[0].plot(inner_list, time_avg, "y", marker='o',
            linewidth=3, markersize=15, label="time_avg")

# x,y轴标签
# axs[0].set_xlabel('inner * 4KB = Data Array Size', fontsize=36)
axs[0].set_ylabel('EXEC_time', fontsize=52)
# 控制图例的形状大小：fontsize控制图例字体大小，markerscale控制scatters形状大小，scatterpoints控制scatters的数量
axs[0].legend(loc=2, fontsize=40, scatterpoints=1)
# 设置 y 轴显示网格线
axs[0].grid(axis='y')
axs[0].set_xticks(inner_list)
axs[0].set_xticklabels(['{:g}'.format(x) for x in inner_list])
# 设置x轴和y轴刻度值的字体大小
axs[0].tick_params(axis='x', labelsize=46)
axs[0].tick_params(axis='y', labelsize=46)
# 设置子图1的y轴刻度值用科学计数法表示
axs[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
axs[0].yaxis.offsetText.set_fontsize(42)
# 设置指数为4
axs[0].ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
# 子图1标题
axs[0].set_title('\n\n\n\n\nExection Time Trends\n', fontsize=58)

# plt.tick_params(labelsize=32)  # 刻度字体大小

axs[1].plot(inner_list, hit_min, "g", marker='^',
            linewidth=3, markersize=15, label="hit_rate_min")
axs[1].plot(inner_list, hit_max, "r", marker='v',
            linewidth=3, markersize=15, label="hit_rate_max")
axs[1].plot(inner_list, hit_avg, "y", marker='o',
            linewidth=3, markersize=15, label="hit_rate_avg")

# x,y轴标签
axs[1].set_ylabel('L2 Hit Rate %', fontsize=52)
# 控制图例的形状大小：fontsize控制图例字体大小，markerscale控制scatters形状大小，scatterpoints控制scatters的数量
axs[1].legend(loc=6, fontsize=40, scatterpoints=1)
# 设置 y 轴显示网格线
axs[1].grid(axis='y')
# inner_list_str as x轴刻度
axs[1].set_xticks(inner_list)
axs[1].set_xticklabels(['{:g}'.format(x) for x in inner_list])
# 设置x轴和y轴刻度值的字体大小
axs[1].tick_params(axis='x', labelsize=46)
axs[1].tick_params(axis='y', labelsize=46)
axs[1].set_title('Hit Rate Trends\n', fontsize=58)

# 子图3：显示原始数据，保留8位小数
cellT = np.around(np.vstack((time_max/10000, time_avg/10000,
                  time_min/10000, inner_list, hit_max, hit_avg, hit_min)), 8)
table = axs[2].table(cellText=cellT,
                     rowLabels=['time_max', 'time_avg', 'time_min', ' ',
                                'hit_max', 'hit_avg', 'hit_min'],  # 行标题
                     loc='center')
table.set_fontsize(44)
table.scale(1, 10)
axs[2].axis('off')  # 隐藏坐标轴和网格线
axs[2].set_title('\n\nRaw Data Table', fontsize=60)
# 将绘图区域与图片边缘的距离都设置为0.02
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.02, top=0.95)
# 添加总标题
plt.suptitle('Data Analysis of GPU : '+GPU_name +
             '                                                                  inner * 4KB = Data Array Size', fontsize=64, y=0.97)

# plt.gcf().autofmt_xdate()
pic_name = PIC_DIR + "data_analysis.jpg"
# 如果图片文件已存在，则删除
if os.path.exists(pic_name):
    os.remove(pic_name)
plt.savefig(pic_name)
