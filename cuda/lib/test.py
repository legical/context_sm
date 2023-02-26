#!/usr/bin/env python3
from cProfile import label
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import csv

# 原始数据
data = np.array([
    [476, 526, 576, 634],
    [2.311, 4.2322, 5.2311, 9.5364],
    [23.4131, 45.3212, 34.13511, 69.1324]
])
print('0!\n')
# 创建一个3行1列的figure对象，每行放一个subplot
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), sharex=True)

# 绘制子图1
axs[0].plot(data[0], data[1], 'bo-')
axs[0].set_title('Subplot 1')
axs[0].set_ylabel('y1')
print('1!\n')
# 绘制子图2
axs[1].plot(data[0], data[2], 'r^-')
axs[1].set_title('Subplot 2')
axs[1].set_ylabel('y2')
print('2!\n')
# 绘制子图3
axs[2].axis('off')  # 隐藏坐标轴和网格线
axs[2].table(cellText=np.around(data, 3),  # 转置并保留三位小数
             rowLabels=['x', 'y1', 'y2'],  # 行标题
             colLabels=['', '', '', ''])  # 列标题

# 调整subplot之间的间距
plt.subplots_adjust(hspace=0.5)

# 显示图形
plt.savefig("t.png")
