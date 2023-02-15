#!/bin/bash

script_dir=$(
    cd $(dirname $0)
    pwd
)
# 3060 or 1070
GPU_name=$(nvidia-smi -q | grep "Product Name" | awk -F ' ' '{print $NF}')
echo "You are running in GPU $GPU_name."

# /home/bric/Workspace/context_sm/cuda/src/dissect_page
proj_dir="$script_dir/../.."
# /home/bric/Workspace/context_sm/cuda

# 判断是否安装了 python 模块
function python_model_check() {
    if python3 -c "import $1" >/dev/null 2>&1; then
        echo "$1 has been installed."
    else
        echo -e "\033[31mInstalling $1.\033[0m"
        python3 -m pip install -U $1
    fi
}

# 检查csv输出目录是否存在
if [ -d $proj_dir/build ]; then
    rm -r $proj_dir/build
fi
mkdir -m 754 $proj_dir/build
# echo -e "\033[34m.csv build directory successfully created.\033[0m"

cd $proj_dir/build
echo -e "\033[34mStart compiling the project......\033[0m"
cmake .. && make

# 检查csv输出目录是否存在
rm -rf $script_dir/data-$GPU_name
mkdir -m 754 $script_dir/data-$GPU_name
mkdir -m 754 $script_dir/data-$GPU_name/log
mkdir -m 754 $script_dir/data-$GPU_name/pic
echo -e "\033[34m.csv & pic file output directory successfully created.\033[0m"

# default : 3060
inner_cycle=426
CUDA_TOOL_DIR="/usr/local/cuda-11.7/bin"
if [ $GPU_name -eq 1070 ]; then
    inner_cycle=362
    CUDA_TOOL_DIR="/usr/local/cuda-11.8/bin"
fi

for ((j = 1; j <= 5; j++)); do
    # 3060 L2 cache: 2359296B = 2.25MB = 576 * 4KB      476 526 576 626 676
    # 1070 L2 cache: 2097152B = 2MB    = 512 * 4KB      412 462 512 562 612
    inner_cycle=$((inner_cycle + 50))
    echo "Starting running kernel in GPU $GPU_name, $inner_cycle * 1024 times."
    echo -e "Index \t Time"
    for ((i = 1; i <= 1024; i++)); do
        if [ $GPU_name -eq 3060 ]; then
            # get sudo right
            echo "0923326" | sudo -S $CUDA_TOOL_DIR/ncu --section MemoryWorkloadAnalysis ./l2_dissect_test $inner_cycle $i $GPU_name | tee -a $script_dir/data-$GPU_name/log/dis-${inner_cycle}.log
            sudo chmod 777 $script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
            tail -n 4 $script_dir/data/log/dis-${inner_cycle}.log | grep "L2 Hit Rate" | awk -F ' ' '{print $NF}' >>$script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
        elif [ $GPU_name -eq 1070 ]; then
            # temp log
            /usr/bin/script -qf data-$GPU_name.log -c "echo 'neu' | sudo -S /usr/local/cuda-11.8/bin/nvprof --metrics l2_tex_hit_rate ./l2_dissect_test $inner_cycle $i $GPU_name"
            # save info line to true log 
            cat data-$GPU_name.log | grep "l2_tex_hit_rate" | tail -n 1 >>$script_dir/data-$GPU_name/log/dis-${inner_cycle}.log
            echo "neu" | sudo -S chmod 777 $script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
            # save hit rate info to csv file
            cat data-$GPU_name.log | grep "l2_tex_hit_rate" | awk -F ' ' '{print $NF}' | tail -n 1 | sed 's/%//g' >>$script_dir/data-$GPU_name/Dissect-inner${inner_cycle}.csv
        else
            echo "Sorry, not support for $GPU_name."
        fi
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
