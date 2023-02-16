#!/bin/bash
# /home/bric/Workspace/context_sm/cuda/src/dissect_page
script_dir=$(
    cd $(dirname $0)
    pwd
)

proj_dir="$script_dir/../.."
# /home/bric/Workspace/context_sm/cuda
source $proj_dir/lib/tool.sh

# 3060 or 1070
check_GPU_Driver
echo "You are running in GPU $GPU_name."

# delete & recreate build 目录
recreate_build
# isGoon
# 检查csv输出目录是否存在
rm -rf $script_dir/data-$GPU_name
mkdir -m 754 $script_dir/data-$GPU_name
mkdir -m 754 $script_dir/data-$GPU_name/log
mkdir -m 754 $script_dir/data-$GPU_name/pic
echo -e "\033[34m.csv & pic file output directory successfully created.\033[0m"

cd $proj_dir/build
echo -e "\033[34mStart compiling the project......\033[0m"

# default : 3060
inner_cycle=426
CUDA_TOOL_DIR="/usr/local/cuda-11.7/bin"
# if GPU name == 1070
if [ $GPU_name -eq 1070 ]; then
    inner_cycle=362
    CUDA_TOOL_DIR="/usr/local/cuda-11.8/bin"
    # contrl code data path = data-1070
    cmake -DGPU_1070_IN=1 .. && make
# else if GPU name == 3060
elif [ $GPU_name -eq 3060 ]; then
    # contrl code data path = data-3060
    cmake -DGPU_1070_IN=0 .. && make
fi

OUT_RUNNING=5
INNER_RUNNING=1024
for ((j = 1; j <= OUT_RUNNING; j++)); do
    # 3060 L2 cache: 2359296B = 2.25MB = 576 * 4KB      476 526 576 626 676
    # 1070 L2 cache: 2097152B = 2MB    = 512 * 4KB      412 462 512 562 612
    inner_cycle=$((inner_cycle + 50))
    echo "Starting running kernel in GPU $GPU_name, $inner_cycle * 1024 times."
    echo -e "Index \t Time"    
    for ((i = 1; i <= INNER_RUNNING; i++)); do
        if [ "$GPU_name" -eq 3060 ]; then
            # get sudo right
            # echo "0923326" | sudo -S $CUDA_TOOL_DIR/ncu --section MemoryWorkloadAnalysis ./l2_dissect_test $inner_cycle $i | tee -a $script_dir/data-$GPU_name/log/dis-${inner_cycle}.log
            echo "0923326" | sudo -S $CUDA_TOOL_DIR/ncu --metrics group:memory__l2_cache_table ./l2_dissect_test $inner_cycle $i >data-$GPU_name.log
            hit_line=$(cat data-$GPU_name.log | grep "lts__t_sectors_lookup_hit.sum" | sed 's/,//g')
            miss_line=$(cat data-$GPU_name.log | grep "lts__t_sectors_lookup_miss.sum" | sed 's/,//g')
            echo -e "No.$j/$OUT_RUNNING : $i \nhit_line: $hit_line\nmiss_line: $miss_line" >>$script_dir/data-$GPU_name/log/dis-${inner_cycle}.log
            hit_num=$(echo $hit_line | awk -F ' ' '{print $NF}')
            miss_num=$(echo $miss_line | awk -F ' ' '{print $NF}')
            sudo chmod 777 $script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
            # tail -n 4 $script_dir/data-$GPU_name/log/dis-${inner_cycle}.log | grep "L2 Hit Rate" | awk -F ' ' '{print $NF}' | sed 's/,//g' >>$script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
            echo "$hit_num,$miss_num" >>$script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
        elif [ "$GPU_name" -eq 1070 ]; then
            # temp log
            /usr/bin/script -qf data-$GPU_name.log -c "echo 'neu' | sudo -S /usr/local/cuda-11.8/bin/nvprof --metrics l2_tex_hit_rate ./l2_dissect_test $inner_cycle $i"
            # save info line to true log
            cat data-$GPU_name.log | grep "l2_tex_hit_rate" | tail -n 1 >>$script_dir/data-$GPU_name/log/dis-${inner_cycle}.log
            echo "neu" | sudo -S chmod 777 $script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
            # save hit rate info to csv file
            cat data-$GPU_name.log | grep "l2_tex_hit_rate" | awk -F ' ' '{print $NF}' | tail -n 1 | sed 's/%//g' >>$script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
        else
            echo "Sorry, not support for $GPU_name."
        fi
        progress_bar $i $INNER_RUNNING $j $OUT_RUNNING
        sleep 0.1
    done
done
rm -rf ./*

# 根据数据画图
echo -e "\n\033[34mDraw line charts according to the output data......\033[0m"
cd $script_dir
# 判断是否安装了 numpy matplotlib 模块
python_model_check numpy
python_model_check matplotlib

python3 draw.py $GPU_name
echo -e "\n\033[34m$script_dir/data-$GPU_name\033[0m to see execution time data."
echo -e "\033[34m$script_dir/data-$GPU_name/pic\033[0m to see line charts."
echo -e "\033[32mALL done."

# git
cd $proj_dir && cd ..
git add .
git commit -a -m "get data of l2_dissect_test of $GPU_name"
git push
