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

cfg="cfg-cv.png"
bbcfg="bbcfg-cv.png"
line="line-cv.txt"

cuobjdump cu_ran_test -xelf cu_ran_test.1.sm_86.cubin
nvdisasm -cfg cu_ran_test.1.sm_86.cubin | dot -o$cfg -Tpng
nvdisasm -bbcfg cu_ran_test.1.sm_86.cubin | dot -o$bbcfg -Tpng
nvdisasm -gi cu_ran_test.1.sm_86.cubin > $line 2>&1

mv $cfg $bbcfg $line $script_dir/info
cd $script_dir && rm -rf build