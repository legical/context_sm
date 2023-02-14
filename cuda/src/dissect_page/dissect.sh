#!/bin/bash

script_dir=$(
    cd $(dirname $0)
    pwd
)
# /home/bric/Workspace/context_sm/cuda/src/dissect_page
proj_dir="$script_dir/../.."
# /home/bric/Workspace/context_sm/cuda

# 检查csv输出目录是否存在
if [ -d $proj_dir/build ]; then
    rm -r $proj_dir/build
fi
mkdir -m 754 $proj_dir/build
# echo -e "\033[34m.csv build directory successfully created.\033[0m"

cd $proj_dir/build
echo -e "\n\033[34mStart compiling the project......\033[0m"
cmake .. && make

# 检查csv输出目录是否存在
if [ -d $script_dir/data ]; then
    chmod 754 $script_dir/data
    echo ".csv file output directory already exists."
else
    mkdir -m 754 $script_dir/data
    echo -e "\033[34m.csv file output directory successfully created.\033[0m"
fi

inner_cycle=448
for ((j = 1; j <= 3; j++)); do
    # 480 512 544
    inner_cycle=$((inner_cycle + 32))
    for ((i = 1; i <= 1024; i++)); do
        ./l2_dissect_test $inner_cycle $i
    done
done
rm -rf ./*
# git
cd $proj_dir && cd ..
git add .
git commit -a -m "get data of l2_dissect_test"
git push