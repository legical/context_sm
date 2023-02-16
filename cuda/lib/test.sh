#!/bin/bash
# \33[2K 删除光标当前所在的整行
# \033[A将光标上移一行，但移至同一列，即不移至该行的开头
# \r将光标移至行首(r用于快退)，但不删除任何内容
GREEN_SHAN='\E[5;32;49;1m' # 亮绿色闪动
RES='\E[0m'                # 清除颜色
for ((j = 1; j <= 5; j++)); do
    # echo "No.2/6 : 266 =============  ] [25%] [/] [No.2/6]"
    echo "hit_line:     lts__t_sectors_lookup_hit.sum 9040"
    echo "miss_line:     lts__t_sectors_lookup_miss.sum 144363"
    sleep 0.1
    printf "\033[A \33[2K"
    echo "****************"
    printf "\r${GREEN_SHAN} Process: %3d ${RES}" "$j"
done
