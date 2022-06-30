#!/bin/bash
#nvidia-smi -c compute_mode -i target_gpu_id
#compute_mode  = 0/Default, 1/Exclusive_Thread, 2/Prohibited, 3/Exclusive_Process
#target_gpu_id = Id number found by running the following command: nvidia-smi -q
#for my device it would be: nvidia-smi -c 3 -i 00000000:0B:00.0
echo -e "***** Note: Firstly chmod +x this script and used as \033[31mroot\033[0m privilege ! *****"
echo -e "***** \033[32mStart\033[0m MPS in \033[32mY\033[0m  or  \033[34mStop\033[0m MPS in \033[34mN\033[0m *****"
read -p "***** Please enter your choice : " ANSWER
case "$ANSWER" in
  [yY] | [yY][eE][sS])
    echo "***** Your choice is : \033[32mStart MPS\033[0m *****"
    export CUDA_VISIBLE_DEVICES="0"
    nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
    nvidia-cuda-mps-control -d
    echo -e "***** \033[32mStart over, exit!\033[0m *****"
    ;;
  [nN] | [nN][oO])
    echo -e "***** Your choice is : \033[34mStop MPS\033[0m *****"
    echo quit | nvidia-cuda-mps-control
    nvidia-smi -i 0 -c DEFAULT
    echo -e "***** \033[34mStop over, exit!\033[0m *****"
    ;;
  *)
    echo -e "***** \033[31mInvalid Choice : Exit!\033[0m *****"
    ;;
esac
