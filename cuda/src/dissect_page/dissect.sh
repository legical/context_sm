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
recreate_build $proj_dir
# isGoon
# 检查csv输出目录是否存在
rm -rf $script_dir/data-$GPU_name
mkdir -m 754 $script_dir/data-$GPU_name
mkdir -m 754 $script_dir/data-$GPU_name/log
mkdir -m 754 $script_dir/data-$GPU_name/pic
echo -e "\033[34m.csv & pic file output directory successfully created.\033[0m"

cd $proj_dir/build
echo -e "\033[34mStart compiling the project......\033[0m"

# if GPU name == 1070
if [ $GPU_name -eq 1070 ]; then
    inner_cycle_list=(412 462 512 563 614 666 717 768 819 870 922 973 1024)
    # CUDA_TOOL_DIR="/usr/local/cuda-11.8/bin"
    CUDA_PREF_TOOL=$(whereis nvprof | awk -F ' ' '{print $2}')
    # contrl code data path = data-1070
    cmake -DGPU_1070_IN=1 .. && make
# else if GPU name == 3060
elif [ $GPU_name -eq 3060 ]; then
    # contrl code data path = data-3060
    inner_cycle_list=(476 526 576 634 691 749 806 864 922 979 1037 1094 1152)
    CUDA_PREF_TOOL=$(whereis ncu | awk -F ' ' '{print $2}')
    cmake -DGPU_1070_IN=0 .. && make
fi

USR_PASSWD=$(whoami)
OUT_RUNNING=${#inner_cycle_list[@]}
INNER_RUNNING=1024
for ((j = 1; j <= $OUT_RUNNING; j++)); do
    # 3060 L2 cache: 2359296B = 2.25MB = 576 * 4KB      476 526 576 626 676 864
    # 1070 L2 cache: 2097152B = 2MB    = 512 * 4KB      412 462 512 562 612 768
    inner_cycle=${inner_cycle_list[${j} - 1]}
    echo "GPU $GPU_name uses $CUDA_PREF_TOOL , Test $inner_cycle * 1024 times."
    # echo -e "Index \t Time"
    for ((i = 1; i <= $INNER_RUNNING; i++)); do
        if [ "$GPU_name" -eq 3060 ]; then
            # get sudo right
            echo $USR_PASSWD | sudo -S $CUDA_PREF_TOOL --metrics group:memory__l2_cache_table ./l2_dissect_test $inner_cycle $i >data-$GPU_name.log
            hit_line=$(cat data-$GPU_name.log | grep "lts__t_sectors_lookup_hit.sum" | sed 's/,//g')
            miss_line=$(cat data-$GPU_name.log | grep "lts__t_sectors_lookup_miss.sum" | sed 's/,//g')
            del_this_line
            echo -e "No.$j/$OUT_RUNNING : $i \nhit_line: $hit_line\nmiss_line: $miss_line" | tee -a $script_dir/data-$GPU_name/log/dis-${inner_cycle}.log
            hit_num=$(echo $hit_line | awk -F ' ' '{print $NF}')
            miss_num=$(echo $miss_line | awk -F ' ' '{print $NF}')
            sudo chmod 777 $script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
            # tail -n 4 $script_dir/data-$GPU_name/log/dis-${inner_cycle}.log | grep "L2 Hit Rate" | awk -F ' ' '{print $NF}' | sed 's/,//g' >>$script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
            echo "$hit_num,$miss_num" >>$script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
        elif [ "$GPU_name" -eq 1070 ]; then
            # temp log
            /usr/bin/script -qf data-$GPU_name.log -c "echo $USR_PASSWD | sudo -S $CUDA_PREF_TOOL --metrics l2_tex_hit_rate ./l2_dissect_test $inner_cycle $i" >/dev/null 2>&1
            # save info line to true log
            hit_line=$(cat data-$GPU_name.log | grep "l2_tex_hit_rate" | tail -n 1 | sed 's/,//g' | sed 's/          1                          //' | sed 's/                   //')
            del_this_line
            echo "No.$j/$OUT_RUNNING | $i:  $hit_line" | tee -a $script_dir/data-$GPU_name/log/dis-${inner_cycle}.log
            echo $USR_PASSWD | sudo -S chmod 777 $script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
            # save hit rate info to csv file
            echo $hit_line | awk -F ' ' '{print $NF}' | sed 's/%//g' >>$script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
        else
            echo "Sorry, not support for $GPU_name. Exit!"
        fi
        progress_bar $i $INNER_RUNNING $j $OUT_RUNNING
        sleep 0.1
    done
    printf "\e[0;35;1m No.$j has done! Total: $INNER_RUNNING \e[0m\n"
done
rm -rf ./*

# 根据数据画图
echo -e "\n\033[34mDraw line charts according to the output data......\033[0m"
cd $script_dir
# 判断是否安装了 numpy matplotlib 模块
python_model_check numpy
python_model_check matplotlib
py3=$(whereis python3 | awk -F ' ' '{print $2}')

$py3 draw.py $GPU_name
echo -e "\n\033[34m$script_dir/data-$GPU_name\033[0m to see execution time data."
echo -e "\033[34m$script_dir/data-$GPU_name/pic\033[0m to see line charts."

# git
cd $proj_dir && cd ..
git add .
git commit -a -m "get data of l2_dissect_test of $GPU_name"
git push
echo -e "\033[32mALL done."
