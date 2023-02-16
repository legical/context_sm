#!/bin/bash
# /home/bric/Workspace/context_sm/cuda/lib
LIB_DIR=$(
    cd $(dirname $0)
    pwd
)

# /home/bric/Workspace/context_sm/cuda
PROJ_DIR="$LIB_DIR/.."

trap 'onCtrlC' INT
function onCtrlC() {
    #捕获CTRL+C，当脚本被ctrl+c的形式终止时同时终止程序的后台进程
    kill -9 ${do_sth_pid} ${progress_pid}
    echo
    echo 'Ctrl+C 中断进程'
    exit 1
}

# 是否继续执行脚本
function isGoon() {
    read -p "Do you want to continue? [y/n] " input

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

function check_GPU_Driver {
    # 3060 or 1070
    GPU_name_line=$(nvidia-smi -q | grep "Product Name")
    if [ $? -ne 0 ]; then
        # nvidia-smi 无输出，证明未安装NVIDIA Driver
        echo -e "***** \033[34mPlease install NVIDIA Driver first !!!\033[0m *****"
    else
        GPU_name=$(echo $GPU_name_line | awk -F ' ' '{print $NF}')
    fi
}

# 检查 build 目录是否存在, recreate build 目录
function recreate_build {
    # 检查 build 目录是否存在
    if [ -d $PROJ_DIR/build ]; then
        rm -r $PROJ_DIR/build
    fi
    mkdir -m 754 $PROJ_DIR/build
    # echo -e "\033[34mbuild directory successfully created.\033[0m"
}

# 命令行进度条，该函数接受3个参数，1-进度的整型数值，2-是总数的整型数值, 3-out running times
function progress_bar {
    if [ -z $1 ] || [ -z $2 ]; then
        echo "[E]>>>Missing parameter. \$1=$1, \$2=$2"
        return
    fi

    pro=$1   # 进度的整型数值
    total=$2 # 总数的整型数值
    if [ $pro -gt $total ]; then
        echo "[E]>>>It's impossible that 'pro($pro) > total($total)'."
        return
    fi
    arr=('|' '/' '-' '\\')
    local percent=$(($pro * 100 / $total))
    #echo "pro=$pro, percent=$percent"
    local str=$(for i in $(seq 1 $percent); do printf '='; done)
    let index=pro%4
    # if $3 not exist
    if [ -z $3 ]; then
        printf "[%-100s] [%d%%] [%c]\r" "$str" "$percent" "${arr[$index]}"
    else
        printf "[%-100s] [%d%%] [%c] No.$3\r" "$str" "$percent" "${arr[$index]}"
    fi
    printf "[%-100s] [%d%%] [%c]\r" "$str" "$percent" "${arr[$index]}"
    if [ $percent -eq 100 ]; then
        echo
    fi
}
# ---下面是测试代码---
# 总数为200
# 进度从1增加到200
# 循环调用ProgressBar函数，直到进度增加到200结束
# Tot=200
# for i in `seq 1 $Tot`;do
#     progress_bar $i $Tot
#     sleep 0.1
# done

# 判断是否安装了 python 模块
function python_model_check() {
    if python3 -c "import $1" >/dev/null 2>&1; then
        echo "$1 has been installed."
    else
        echo -e "\033[31mInstalling $1.\033[0m"
        python3 -m pip install -U $1
    fi
}
# ---下面是测试代码---
# 判断是否安装了 numpy matplotlib 模块
# python_model_check numpy
# python_model_check matplotlib

function python_model_check() {
    if python3 -c "import $1" >/dev/null 2>&1; then
        echo "$1 has been installed."
    else
        echo -e "\033[31mInstalling $1.\033[0m"
        python3 -m pip install -U $1
    fi
}
