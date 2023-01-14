#!/bin/bash
script_dir=$(
    cd $(dirname $0)
    pwd
)

# /home/bric/Workspace/context_sm/cuda
cd $script_dir
# 检查 cmake build 目录是否存在
if [ -d build ]; then
    rm -r build
    # echo "Cmake build directory already exists."
fi
mkdir -m 754 build
echo -e "\033[34mCmake build directory successfully created.\033[0m"

if [ -d info ]; then
    chmod 754 info
    # echo "Cmake build directory already exists."
fi
mkdir -m 754 info

# 进入 build, 生成项目
cd build
echo -e "\n\033[34mStart compiling the project......\033[0m"
cmake .. && make

# 输出每个项目的汇编代码
pro_name=("cu_ran_test" "cu_ran_fork" "vector_add")
mkdir $script_dir/info/sass-ptx
for name in ${pro_name[@]}; do
    cuobjdump ${name} -sass -ptx >${name}.txt 2>&1
    mv -f ${name}.txt $script_dir/info/sass-ptx
done

cfg="cfg-v.png"
bbcfg="bbcfg-v.png"
line="line-v.txt"
proname="cu_ran_test"
# proname="vector_add"

cuobjdump $proname -xelf ${proname}.1.sm_86.cubin
nvdisasm -cfg ${proname}.1.sm_86.cubin | dot -o$cfg -Tpng
nvdisasm -bbcfg ${proname}.1.sm_86.cubin | dot -o$bbcfg -Tpng
nvdisasm -gi ${proname}.1.sm_86.cubin >$line 2>&1

mv -f $cfg $bbcfg $line $script_dir/info
cd $script_dir && rm -rf build

# git
cd ..
git add .
git commit -a -m "get info of $proname"
git push