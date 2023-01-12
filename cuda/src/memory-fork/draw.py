#!/usr/bin/env python3
from cProfile import label
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

# example: python3 draw.py


def readname():
    filePath = "./output/"
    name = os.listdir(filePath)
    csv = []
    for file in name:
        if ".csv" in file:
            csv.append(file)
    return csv


# def get_data(filename, IDlist, EXEClist, minax, GPU_addr_list):
#     IDlist, EXEClist = np.loadtxt(
#         filename, dtype=float, delimiter=',', skiprows=1, usecols=(0, 1), unpack=True)
#     IDlist = np.array(IDlist)
#     minax.append(EXEClist.min())
#     minax.append(EXEClist.max())
#     minax.append(np.mean(EXEClist))
#     addr_list, GPU_addr_list = np.loadtxt(
#         filename, dtype=str, delimiter=',', skiprows=1, usecols=(2, 3), unpack=True)
#     minax.append(len(set(addr_list)))
#     minax.append(len(set(GPU_addr_list)))


csvlist = readname()
# print(csvlist)
for file in csvlist:
    filename = './output/'+file
    # print("\n",filename)
    # 图片dpi=220，尺寸宽和高，单位为英寸
    fig = plt.figure(dpi=220, figsize=(80, 32))
    ax1 = fig.add_subplot(111)

    # 获取ID
    minax = []
    # IDlist, EXEClist, GPU_addr_list, minax = [], [], [], []
    # minax = [min, max, avg, CPU_addr_num, GPU_addr_num]
    # get_data(filename, IDlist, EXEClist, minax, GPU_addr_list)
    IDlist, EXEClist = np.loadtxt(
        filename, dtype=float, delimiter=',', skiprows=1, usecols=(0, 1), unpack=True)
    minax.append(EXEClist.min())
    minax.append(EXEClist.max())
    minax.append(np.mean(EXEClist))
    addr_list, GPU_addr_list = np.loadtxt(
        filename, dtype=str, delimiter=',', skiprows=1, usecols=(2, 3), unpack=True)
    minax.append(len(set(addr_list)))
    minax.append(len(set(GPU_addr_list)))

    # 获取平均值
    avg_x = [1, IDlist[-1]]
    avg_y = [minax[2], minax[2]]

    ax1.plot(IDlist, EXEClist, "g", marker='D',
             markersize=5, label="Execution time")
    ax1.plot(avg_x, avg_y, color='r', label="Avg_time")

    # y轴刻度值
    # plt.yticks(np.arange(min_y_lable, max_y_lable, 0.2))
    ax1.set_ylim((np.floor(minax[0]), np.ceil(minax[1])))
    ax1.set_xlabel('Index', fontsize=36)
    ax1.set_ylabel('EXEC_time', fontsize=36)
    # 控制图例的形状大小：fontsize控制图例字体大小，markerscale控制scatters形状大小，scatterpoints控制scatters的数量
    ax1.legend(loc=4, fontsize=32, scatterpoints=1)
    # 设置 y 轴显示网格线
    ax1.grid(axis='y')

    ax2 = ax1.twinx()  # 创建第二个坐标轴，GPU 地址
    ax2.plot(IDlist, GPU_addr_list, 'o-', c='blue',
             markersize=5, label="GPU address", linewidth=0.4)
    ax2.set_ylabel('GPU_addr', fontsize=36)
    ax2.set_yticks([GPU_addr_list[0], GPU_addr_list[-1]])
    # 控制图例的形状大小：fontsize控制图例字体大小，markerscale控制scatters形状大小，scatterpoints控制scatters的数量
    ax2.legend(loc=1, fontsize=32, scatterpoints=1)
    ax2.grid(visible=False)

    plt.tick_params(labelsize=32)  # 刻度字体大小
    # plt.title('执行时间折线图')  # 折线图标题
    chart_title = 'min={}   max={}   avg={} mS   ||  times={}   addr_num={}   GPU_addr_num={}'
    plt.title(chart_title.format(
        minax[0], minax[1], minax[2], int(IDlist[-1]), minax[3], minax[4]), fontsize=46)
    # plt.gcf().autofmt_xdate()
    filename = filename.replace("./output/", "./output/pic/", 1)
    pic_name = filename.replace("csv", "jpg", 1)
    # 如果图片文件已存在，则删除
    if os.path.exists(pic_name):
        os.remove(pic_name)
    plt.savefig(pic_name)
    # plt.show()
