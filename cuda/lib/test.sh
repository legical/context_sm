#!/bin/bash

GREEN_SHAN='\E[5;32;49;1m' # 亮绿色闪动
RES='\E[0m'                # 清除颜色
for ((j = 1; j <= 5; j++)); do
    # echo "No.2/6 : 266 =============  ] [25%] [/] [No.2/6]"
    echo "hit_line:     lts__t_sectors_lookup_hit.sum 9040"
    echo "miss_line:     lts__t_sectors_lookup_miss.sum 144363"
    sleep 0.1
    printf "\b\r\b\r\b\r"
    echo "****************"
    printf "\r${GREEN_SHAN} Process: %3d ${RES}" "$j"
done
