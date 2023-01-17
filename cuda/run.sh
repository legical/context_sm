#!/bin/bash

# example: ./run.sh mode=2 EXEC_time=1000 ARR_size=1G
# mode1:单个程序里有多个循环    mode2:多次运行一个程序
script_dir=$(
    cd $(dirname $0)
    pwd
)
# /home/bric/Workspace/context_sm/cuda

dir="$script_dir/src"
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

# 删除运行过程中生成的0 1 2 文件
function del_extra_file() {
    cd $script_dir && rm -rf 0 1 2 build
    cd $script_dir/src/memory && rm -rf 0 1 2
    cd output && rm -rf 0 1 2
    cd $script_dir/src/memory-fork && rm -rf 0 1 2
    cd output && rm -rf 0 1 2
}

# 获取参数
time=1000
size=1073741824
# 1-单词程序多循环 2-多次程序
mode=1
run_dir="memory"

if [ $# ] >0; then
    mode=$1
fi
if [ $# ] >1; then
    time=$2
fi
if [ $# ] >2; then
    size=$3
fi
echo -e "\nShell: You have entered $# parameter."
echo "EXEC_TIMES: $time       ARR_SIZE: ${size} "

if [ $mode = "1" ]
then
	echo "You choose Mode: Loops in a single execution."
    run_dir="memory"
else
    echo "You choose Mode: Execute a single program multiple times."
    run_dir="memory-fork"
fi
# isGoon

# 检查csv输出目录是否存在
if [ -d $dir/$run_dir/output ]; then
    chmod 754 $dir/$run_dir/output
    echo ".csv file output directory already exists."
else
    mkdir -m 754 $dir/$run_dir/output
    echo -e "\033[34m.csv file output directory successfully created.\033[0m"
fi
# 检查pic输出目录是否存在
if [ -d $dir/$run_dir/output/pic ]; then
    chmod 754 $dir/$run_dir/output/pic
    echo ".jpg file output directory already exists."
else
    mkdir -m 754 $dir/$run_dir/output/pic
    echo -e "\033[34m.jpg file output directory successfully created.\033[0m"
fi

# /home/bric/Workspace/context_sm/cuda
cd $script_dir
# 检查 cmake build 目录是否存在
if [ -d build ]; then
    rm -r build
    # echo "Cmake build directory already exists."
fi
mkdir -m 754 build
echo -e "\033[34mCmake build directory successfully created.\033[0m"

# 进入 build, 生成项目
cd build
echo -e "\n\033[34mStart compiling the project......\033[0m"
cmake .. && make

# isGoon

# 执行项目
echo -e "\n\n\033[34mStart running the project for $time times......\033[0m"
if [ $mode = "1" ]
then
	./cu_ran $time $size
else
    filename_date=$(date "+%d%H%M")
    # echo "filename_date is $filename_date"
    for ((i=1; i<=$time; i++))
    do
        ./cu_ran_fork $i $time "$filename_date"
    done
fi

# 根据数据画图
echo -e "\n\033[34mDraw line charts according to the output data......\033[0m"
cd $dir/$run_dir
# 判断是否安装了 numpy matplotlib 模块
python_model_check numpy
python_model_check matplotlib

python3 draw.py
echo -e "\n\033[34m$dir/$run_dir/output\033[0m to see execution time data."
echo -e "\033[34m$dir/$run_dir/output/pic\033[0m to see line charts."
del_extra_file
echo -e "\033[32mALL done."