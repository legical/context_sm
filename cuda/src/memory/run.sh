#!/bin/bash
script_dir=$(
    cd $(dirname $0)
    pwd
)
# /home/bric/Workspace/context_sm/cuda/src/memory

dir=$(dirname $script_dir)
# /home/bric/Workspace/context_sm/cuda/src

# 是否继续执行脚本
function isGoon() {
    read -p "Are you sure to continue? [y/n] " input

    case $input in
    [yY]*) ;;

    [nN]*)
        exit
        ;;
    *)
        echo -e "Just enter y or n, please.\n"
        exit
        ;;
    esac
}

# 判断是否安装了 python 模块
function python_model_check() {
    if python -c "import $1" >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# 检查csv输出目录是否存在
if [ -d $script_dir/output ]; then
    chmod 754 $script_dir/output
    echo ".csv file output directory already exists."
else
    mkdir -m 754 $script_dir/output
    echo ".csv file output directory successfully created."
fi
# 检查pic输出目录是否存在
if [ -d $script_dir/output/pic ]; then
    chmod 754 $script_dir/output/pic
    echo ".jpg file output directory already exists."
else
    mkdir -m 754 $script_dir/output/pic
    echo ".jpg file output directory successfully created."
fi

# 获取参数
time="1000"
size="1073741824"

if [ $# ] >0; then
    time=$1
fi
if [ $# ] >1; then
    size=$2
fi
# echo -e "\nShell: You have entered $# parameter."
# echo "EXEC_TIMES: $time       ARR_SIZE: $size "

# isGoon

# /home/bric/Workspace/context_sm/cuda
cd $(dirname $dir)
# 检查 cmake build 目录是否存在
if [ -d build ]; then
    chmod 754 build
    echo "Cmake build directory already exists."
else
    mkdir -m 754 build
    echo "Cmake build directory successfully created."
fi

# 进入 build, 生成项目
cd build
echo -e "\nStart compiling the project......"
cmake .. && make

# 执行项目
echo -e "\n\nStart running the project......"
./cu_ran $time $size

# 根据数据画图
echo -e "\nDraw line charts according to the output data......"
cd $script_dir
# 判断是否安装了 matplotlib 模块
function python_model_check() {
    if python3 -c "import $1" >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}
python_model_check matplotlib
if [ $? == 1 ]; then
    echo "Matplotlib has been installed."
else
    echo "Installing matplotlib."
    python3 -m pip install -U matplotlib
fi

python3 draw.py
echo -e "\n$script_dir/output to see execution time data."
echo "$script_dir/output/pic to see line charts."
echo "ALL done."
