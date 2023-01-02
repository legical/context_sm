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
        echo -e "\033[31mJust enter y or n, please.\033[0m\n"
        exit
        ;;
    esac
}

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
if [ -d $script_dir/output ]; then
    chmod 754 $script_dir/output
    echo ".csv file output directory already exists."
else
    mkdir -m 754 $script_dir/output
    echo -e "\033[34m.csv file output directory successfully created.\033[0m"
fi
# 检查pic输出目录是否存在
if [ -d $script_dir/output/pic ]; then
    chmod 754 $script_dir/output/pic
    echo ".jpg file output directory already exists."
else
    mkdir -m 754 $script_dir/output/pic
    echo -e "\033[34m.jpg file output directory successfully created.\033[0m"
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
    echo -e "\033[34mCmake build directory successfully created.\033[0m"
fi

# 进入 build, 生成项目
cd build
rm -r ./*
echo -e "\n\033[34mStart compiling the project......\033[0m"
cmake .. && make

# 执行项目
echo -e "\n\n\033[34mStart running the project......\033[0m"
./cu_ran $time $size

# 根据数据画图
echo -e "\n\033[34mDraw line charts according to the output data......\033[0m"
cd $script_dir
# 判断是否安装了 numpy matplotlib 模块
python_model_check numpy
python_model_check matplotlib

python3 draw.py
echo -e "\n\033[34m$script_dir/output\033[0m to see execution time data."
echo -e "\033[34m$script_dir/output/pic\033[0m to see line charts."
echo -e "\033[32mALL done."
