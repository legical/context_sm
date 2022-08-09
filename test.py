from cProfile import label
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

# def find_kernelnum(filename):
#     # filename = 'globa-outdata-k42-s162142-b6.csv'
#     index = filename.find('-k')+2
#     kernel = ''
#     while index < len(filename):
#         if filename[index] == '-':
#             break
#         kernel += filename[index]
#         index += 1
#     return int(kernel)

# k=find_kernelnum('globa-outdata-k42-s162142-b6.csv')
# print(k)
nums=[40,36,89,2,36,100,7,2,5,-20.5,-999]
def list_all_index(list,v):
    # value所有索引的位置
    v_list = []
    # value第一次出现的位置
    i = list.index(v)+1
    v_list.append(i-1)
    while i < len(list):
        if list[i]==v:
            v_list.append(i)
        i+=1
    return v_list
#检索列表中的所有元素

print(list_all_index(nums,2))