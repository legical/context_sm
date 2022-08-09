from cProfile import label
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

def find_kernelnum(filename):
    # filename = 'globa-outdata-k42-s162142-b6.csv'
    index = filename.find('-k')+2
    kernel = ''
    while index < len(filename):
        if filename[index] == '-':
            break
        kernel += filename[index]
        index += 1
    return int(kernel)

k=find_kernelnum('globa-outdata-k42-s162142-b6.csv')
print(k)