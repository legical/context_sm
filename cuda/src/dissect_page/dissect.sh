#!/bin/bash

script_dir=$(
    cd $(dirname $0)
    pwd
)
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
rm -rf $script_dir/data
mkdir -m 754 $script_dir/data
mkdir -m 754 $script_dir/data/log
mkdir -m 754 $script_dir/data/pic
echo -e "\033[34m.csv & pic file output directory successfully created.\033[0m"

inner_cycle=426
for ((j = 1; j <= 5; j++)); do
    # 3060 L2 cache: 2359296B = 2.25MB = 576 * 4KB
    # 476 526 576 626 676
    inner_cycle=$((inner_cycle + 50))
    echo " Starting running kernel, $inner_cycle * 1024 times."
    echo -e "Index \t Time"
    for ((i = 1; i <= 1024; i++)); do
        # get sudo right
        echo "0923326" | sudo -S /usr/local/cuda-11.7/bin/ncu --section MemoryWorkloadAnalysis ./l2_dissect_test $inner_cycle $i | tee -a $script_dir/data/log/dis-${inner_cycle}.log
        sudo chmod 777 $script_dir/data/Dissect-inner${inner_cycle}.csv
        tail -n 4 $script_dir/data/log/dis-${inner_cycle}.log | grep "L2 Hit Rate" | awk -F ' ' '{print $NF}' >>$script_dir/data/Dissect-inner${inner_cycle}.csv
        # ./l2_dissect_test $inner_cycle $i
    done
done
rm -rf ./*

# 根据数据画图
echo -e "\n\033[34mDraw line charts according to the output data......\033[0m"
cd $script_dir
# 判断是否安装了 numpy matplotlib 模块
python_model_check numpy
python_model_check matplotlib

python3 draw.py
echo -e "\n\033[34m$script_dir/data\033[0m to see execution time data."
echo -e "\033[34m$script_dir/data/pic\033[0m to see line charts."
echo -e "\033[32mALL done."

# git
cd $proj_dir && cd ..
git add .
git commit -a -m "get data of l2_dissect_test in 3060"
git push
