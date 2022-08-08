from cProfile import label
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

# run: python drawpic.py [filename] [kernelnums]
# example: python drawpic.py ./outdata/share-outdata-s16121412-b6.csv 4


def max_sm(sm_max_list):
    temp = 0
    for sm_max in sm_max_list:
        if temp < sm_max:
            temp = sm_max
    return temp+1


def get_data(filename, smids, start_times, end_times, kerID, time_limit, sm_max_list):
    # '''get the highs and lows from a data file'''
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        for row in reader:
            if(int(row[0]) == kerID):
                try:
                    smid = int(row[4])
                    start_time = float(row[5])
                    end_time = float(row[6])
                    if(len(time_limit) == 0):
                        time_limit.append(start_time)
                        time_limit.append(end_time)
                    elif (time_limit[0] > start_time):
                        time_limit[0] = start_time
                    elif (time_limit[1] < end_time):
                        time_limit[1] = end_time
                        # print("now min_time is : ",time_limit[0],"\tmax_time is: ",time_limit[1])
                except ValueError:
                    print(smid, 'reading data error!\n')
                else:
                    if smids.count(smid) < 1 or start_times.count(start_time) < 1:
                        smids.append(smid)
                        start_times.append(start_time)
                        end_times.append(end_time)
                        if sm_max_list[kerID] < smid:
                            sm_max_list[kerID] = smid
            else:
                continue


filename = sys.argv[1]
kernelnums = int(sys.argv[2])

# 蓝色实心圈,洋红色点标记,绿色倒三角,黄色上三角,红色+,黑色正方形,青绿色菱形,白色x
# line_style = ['bo', 'm+', 'gv', 'y^', 'rx', 'ks', 'cD']
line_style = ['b', 'm', 'g', 'y', 'r', 'k', 'c']
# 图片dpi=220，尺寸宽和高，单位为英寸
fig = plt.figure(dpi=220, figsize=(48, 32))

# 获取各个kernel的sm数目
sm_max_list = []
# 获取时间上下限 0-下限 1-上限
time_limit = []
kernel_index = 0
while(kernel_index < kernelnums):
    # 获取每个kernel的数据
    smids, start_times, end_times = [], [], []
    sm_max_list.append(0)
    get_data(filename, smids, start_times, end_times,
             kernel_index, time_limit, sm_max_list)
    # print(smids)
    # print("smids_lenth is",len(smids))
    # 绘图，只从开始-结束时间绘图
    kernel_data_index = 0
    while(kernel_data_index < len(start_times)):
        xpoints = np.array([start_times[kernel_data_index]+20,
                           end_times[kernel_data_index]])
        ypoints = np.array([smids[kernel_data_index]*kernelnums+kernel_index,
                           smids[kernel_data_index]*kernelnums+kernel_index])

        # print("xpoints is ", xpoints, "\typoints is: ", ypoints)
        plt.plot(xpoints, ypoints,
                 line_style[kernel_index], label=str(kernel_index)+',sm:'+str(sm_max_list[kernel_index]+1),  marker='o', linewidth=1)
        # r'$xxxx$'
        # xy=蓝色点位置
        # xytext：描述框相对xy位置
        # textcoords='offset points'，以xy为原点偏移xytext
        # arrowprops = 画弧线箭头，'---->', rad=.2-->0.2弧度
        plt.annotate('kernel' + str(kernel_index)+', '+str(xpoints[0]-25), xy=(xpoints[0], ypoints[0]), xytext=(+10, +10), textcoords='offset points', fontsize=16,
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        # 绘制下一个开始-结束时间线
        kernel_data_index += 1
    # 绘制下一个kernel的数据
    kernel_index += 1
max_sm_num = max_sm(sm_max_list)
print(sm_max_list)
print("max_sm is ", max_sm_num, "\tmin_time is ",
      time_limit[0], "\tmax_time is: ", time_limit[1])
# Format plot
title = 'Distribution on the SM of each kernel | MPS'
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.title(title, fontsize=24)
plt.xlabel('EXEC_time', fontsize=16)
plt.xlim(math.floor(time_limit[0])-200, math.ceil(time_limit[1])+200)  # 设置x轴范围
fig.autofmt_xdate()  # 绘制斜的日期标签
plt.ylabel('SMID', fontsize=16)
# 设置 y 轴显示网格线
plt.grid(axis='y')
# 设置 y 轴标签
y_index = 0
ynum, y_smstr = [], []
while y_index < max_sm_num:
    num = (y_index+1) * kernelnums - 1
    smstr = 'sm'+str(y_index)
    ynum.append(num)
    y_smstr.append(smstr)
    y_index += 1
plt.yticks(ynum, y_smstr)
plt.tick_params(axis='both', labelsize=16)

plt.ylim(-1, kernelnums*max_sm_num)  # 设置y轴范围
filename = filename.replace("./outdata/", "./outdata/pic/", 1)
pic_name = filename.replace("csv", "jpg", 1)
# 如果图片文件已存在，则删除
if os.path.exists(pic_name):
    os.remove(pic_name)
plt.savefig(pic_name)
# plt.show()
