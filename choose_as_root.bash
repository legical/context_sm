#!/bin/bash
#nvidia-smi -c compute_mode -i target_gpu_id
#compute_mode  = 0/Default, 1/Exclusive_Thread, 2/Prohibited, 3/Exclusive_Process
#target_gpu_id = Id number found by running the following command: nvidia-smi -q
#for my device it would be: nvidia-smi -c 3 -i 00000000:0B:00.0
echo "Note: This script must be performed with root privilege !"
echo "Start MPS in Y  or  Stop MPS in N"
read -p "Please enter your choice : " ANSWER
case "$ANSWER" in
  [yY] | [yY][eE][sS])
    echo "Your choice is : Start MPS"
    export CUDA_VISIBLE_DEVICES="0"
    nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
    nvidia-cuda-mps-control -d
    echo "Start over, exit!"
    ;;
  [nN] | [nN][oO])
    echo "Your choice is : Stop MPS"
    echo quit | nvidia-cuda-mps-control
    nvidia-smi -i 0 -c DEFAULT
    echo "Stop over, exit!"
    ;;
  *)
    echo "Invalid Choice : Exit!"
    ;;
esac
