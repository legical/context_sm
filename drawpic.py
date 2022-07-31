import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

# run: python drawpic.py [filename] [kernelnums]
# example: python drawpic.py ./outdata/outdata-s4462-b8.csv 4

def get_data(filename, smids, start_times, end_times, kerID, time_limit):
    # '''get the highs and lows from a data file'''
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        for row in reader:
            if(row[0] == kerID):
                try:
                    smid = int(row[1])
                    start_time = float(row[5])
                    end_time = float(row[6])
                    if (time_limit[0] == 0.0 or time_limit[0] > start_time) :
                        time_limit[0] = start_time
                    if (time_limit[1] == 0.0 or time_limit[1] < end_time) :
                        time_limit[1] = end_time
                    print("now min_time is : ",time_limit[0],"\tmax_time is: ",time_limit[1])
                except ValueError:
                    print(smid, 'reading data error!\n')
                else:
                    smids.append(smid)
                    start_times.append(start_time)
                    end_times.append(end_time)
            else:
                continue
            
filename = sys.argv[1]
kernelnums = int(sys.argv[2])
# 蓝色实心圈,洋红色点标记,绿色倒三角,黄色上三角,红色+,黑色正方形,青绿色菱形,白色x
line_style = ['bo','m.','gv','y^','r+','ks','cD','wx']
# 图片dpi=220，尺寸宽和高，单位为英寸
fig = plt.figure(dpi=220, figsize=(15,9))
# 获取时间上下限 0-下限 1-上限
time_limit = [0.0,0.0]
kernel_index = 0
while(kernel_index < kernelnums):
    # 获取每个kernel的数据
    smids, start_times, end_times = [], [], []
    get_data(filename, smids, start_times, end_times, kernel_index, time_limit)
    # 绘图，只从开始-结束时间绘图
    kernel_data_index = 0
    while(kernel_data_index < len(start_times)):
        xpoints = np.array([start_times[kernel_data_index], smids[kernel_data_index]])
        ypoints = np.array([end_times[kernel_data_index], smids[kernel_data_index]])    
        plt.plot(xpoints, ypoints, line_style[kernel_index])
        # 绘制下一个开始-结束时间线
        kernel_data_index+=1
    # 绘制下一个kernel的数据
    kernel_index += 1
print("min_time is ",time_limit[0],"\tmax_time is: ",time_limit[1])
# Format plot
title = 'Distribution on the SM of each kernel | MPS'
plt.title(title, fontsize=24)
plt.xlabel('EXEC_time', fontsize=16)
plt.xlim(time_limit[0], time_limit[1]) # 设置x轴范围
fig.autofmt_xdate()  # 绘制斜的日期标签
plt.ylabel('SMID', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.ylim(-1, kernelnums) # 设置y轴范围
pic_name = filename.replace("csv", "jpg", 1)
os.remove(pic_name)
plt.savefig(pic_name)
plt.show()