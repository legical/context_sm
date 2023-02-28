#!/bin/bash
# /home/bric/Workspace/context_sm/cuda/lib
LIB_DIR=$(
    cd $(dirname $0)
    pwd
)

# /home/bric/Workspace/context_sm/cuda
# PROJ_DIR="$LIB_DIR/.."

COLOR_FG="\e[30m"
COLOR_BG="\e[42m"
RESTORE_FG="\e[39m"
RESTORE_BG="\e[49m"

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
function recreate_build() {
    # 检查 build 目录是否存在
    PROJ_DIR=$1
    if [ -d $PROJ_DIR/build ]; then
        rm -r $PROJ_DIR/build
    fi
    mkdir -m 754 $PROJ_DIR/build
    # echo -e "\033[34mbuild directory successfully created.\033[0m"
}

# parameter: line numbers want to delete
function del_line() {
    # if do not input parameter
    if [ -z $1 ]; then
        del_num=1
    else
        del_num=$1
    fi
    for ((j = 0; j < del_num; j++)); do
        # \33[2K 删除光标当前所在的整行
        # \033[A将光标上移一行，但移至同一列，即不移至该行的开头
        # \r将光标移至行首(r用于快退)，但不删除任何内容
        printf "\r \33[2K \033[A"
    done
}

function del_this_line() {
    printf "\33[2K"
}
# 打印一个重复 $1字符串 $2次 的新字符串
# printf_new '-' 10 输出一个长度为10的-字符字符串
function printf_new() {
    str=$1
    num=$2
    v=$(printf "%-${num}s" "$str")
    echo -ne "${v// /$str}"
}

# 命令行进度条，该函数接受3个参数，1-进度的整型数值，2-是总数的整型数值, 3-out running times
function progress_bar {
    if [ -z $1 ] || [ -z $2 ]; then
        echo "[E]>>>Missing parameter. \$1=$1, \$2=$2"
        return
    fi

    pro=$1                     # 进度的整型数值
    total=$2                   # 总数的整型数值
    if [ $pro -gt $total ]; then
        echo "[E]>>>It's impossible that 'pro($pro) > total($total)'."
        return
    fi

    GREEN_SHAN='\E[5;32;49;1m' # 亮绿色闪动
    RES='\E[0m'                # 清除颜色
    local cols=$(tput cols)    # 获取列长度
    local color="${COLOR_FG}${COLOR_BG}"
    arr=('|' '/' '-' '\\')
    let index=pro%4
    if [ $cols -le 30 ]; then
        printf "Testing... ${GREEN_SHAN}[%c]${RES}\r" "${arr[$index]}"
        return
    fi

    if [ -z $3 ] || [ -z $4 ]; then
        PRE_STR=$(echo -ne "Test No.${1}/${2}")
    else
        PRE_STR=$(echo -ne "Test ${3}/${4} No.${1}")
    fi

    let bar_size=$cols-27
    let complete_size=$(($pro * $bar_size / $total))
    let remainder_size=$bar_size-$complete_size

    progress_bar=$(echo -ne "["; echo -en "${color}"; printf_new "#" $complete_size; echo -en "${RESTORE_FG}${RESTORE_BG}"; printf_new "." $remainder_size; echo -ne "]");
    #echo "pro=$pro, percent=$percent"

    # Print progress bar
    printf "${PRE_STR}${progress_bar}"
    # printf "\33[2K"

    if [ $pro -eq $total ]; then
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
