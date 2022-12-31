from cProfile import label
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

# example: python draw.py


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
    # fig = plt.figure(dpi=220, figsize=(48, 32))

    # 获取ID
    IDlist, EXEClist = [], []
    minax = [99999999.0,0.0]
    get_data(filename, IDlist, EXEClist, minax)

    plt.plot(IDlist, EXEClist, "g", marker='D', markersize=5, label="执行时间")

    # y轴刻度值
    min_y_lable = np.floor(minax[0])-10
    max_y_lable = np.ceil(minax[1])+2
    plt.yticks(np.arange(min_y_lable, max_y_lable, 0.2))

    # plt.title('执行时间折线图')  # 折线图标题
    plt.xlabel('ID', fontsize=16)
    plt.ylabel('EXEC_time', fontsize=16)
    # 设置 y 轴显示网格线
    plt.grid(axis='y')
    print('min/max/min_y_lable/max_y_lable : ',
          minax[0], minax[1], min_y_lable, max_y_lable, '\n')
    filename = filename.replace("./output/", "./output/pic/", 1)
    pic_name = filename.replace("csv", "jpg", 1)
    # 如果图片文件已存在，则删除
    if os.path.exists(pic_name):
        os.remove(pic_name)
    plt.savefig(pic_name)
    # plt.show()
