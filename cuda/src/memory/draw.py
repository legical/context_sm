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


def get_data(filename, IDlist, EXEClist, minax):
    # '''get the highs and lows from a data file'''
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        for row in reader:
            try:
                IDlist.append(float(row[0]))
                time = float(row[1])
                minax[2] += time
                EXEClist.append(time)
                if minax[0] > time:
                    minax[0] = time
                if minax[1] < time:
                    minax[1] = time
            except ValueError:
                print(row[0], 'reading data error!\n')


csvlist = readname()
# print(csvlist)
for file in csvlist:
    filename = './output/'+file
    # print("\n",filename)
    # 图片dpi=220，尺寸宽和高，单位为英寸
    fig = plt.figure(dpi=220, figsize=(64, 32))

    # 获取ID
    IDlist, EXEClist = [], []
    minax = [99999999.0, 0.0, 0.0]
    get_data(filename, IDlist, EXEClist, minax)

    # 获取平均值
    minax[2] /= IDlist[-1]
    avg_x = [1, IDlist[-1]]
    avg_y = [minax[2], minax[2]]

    plt.plot(IDlist, EXEClist, "g", marker='D',
             markersize=5, label="Execution time")
    plt.plot(avg_x, avg_y, color='r', label="Avg_time")

    # y轴刻度值
    min_y_lable = np.floor(minax[0])-1
    max_y_lable = np.ceil(minax[1])+1
    # plt.yticks(np.arange(min_y_lable, max_y_lable, 0.2))
    plt.ylim((min_y_lable, max_y_lable))

    # plt.title('执行时间折线图')  # 折线图标题
    chart_title = 'min={}    max={}    avg={}'
    plt.title(chart_title.format(minax[0], minax[1], minax[2]), fontsize=42)
    plt.xlabel('ID', fontsize=32)
    plt.ylabel('EXEC_time', fontsize=32)
    plt.tick_params(labelsize=28)  # 刻度字体大小
    # 控制图例的形状大小：fontsize控制图例字体大小，markerscale控制scatters形状大小，scatterpoints控制scatters的数量
    plt.legend(loc=4, fontsize=26, scatterpoints=1)
    # 设置 y 轴显示网格线
    plt.grid(axis='y')
    # print('min/max/min_y_lable/max_y_lable : ',
    #       minax[0], minax[1], min_y_lable, max_y_lable, '\n')
    filename = filename.replace("./output/", "./output/pic/", 1)
    pic_name = filename.replace("csv", "jpg", 1)
    # 如果图片文件已存在，则删除
    if os.path.exists(pic_name):
        os.remove(pic_name)
    plt.savefig(pic_name)
    # plt.show()
