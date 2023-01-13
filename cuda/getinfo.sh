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

# 进入 build, 生成项目
cd build
echo -e "\n\033[34mStart compiling the project......\033[0m"
cmake .. && make

