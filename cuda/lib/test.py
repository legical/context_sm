#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# 原始数据
data = np.array([
    [406, 526, 576, 734],
    [2.311, 4.2322, 5.2311, 9.5364],
    [23.4131, 45.3212, 34.13511, 69.1324]
])
print('0!\n')
# 创建一个3行1列的figure对象，每行放一个subplot
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))

axs[0].plot(data[0], data[1], 'ro-')
axs[0].set_title('Subplot 1')
axs[0].set_xticks(data[0])
axs[0].set_xticklabels(['{:g}'.format(x) for x in data[0]])
axs[0].set_xlabel('x')
axs[0].set_ylabel('y1')

# 子图2：x轴使用第一行数据，y轴使用第三行数据
axs[1].plot(data[0], data[2], 'b^-')
axs[1].set_title('Subplot 2')
axs[1].set_xticks(data[0])
axs[1].set_xticklabels(['{:g}'.format(x) for x in data[0]])
axs[1].set_xlabel('x')
axs[1].set_ylabel('y2')

# 子图3：显示原始数据
cellT = np.around(np.vstack((data[0], data[1], data[2]/10)), 3)
table = axs[2].table(cellText=cellT,  # 保留三位小数
             rowLabels=['x', 'y1', 'y2'],  # 行标题
             loc='center')  
table.set_fontsize(14)
# table.scale(1, 2)
axs[2].axis('off')  # 隐藏坐标轴和网格线

# 显示图形
plt.savefig("t.png")
