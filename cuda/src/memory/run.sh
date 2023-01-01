#!/bin/bash
script_dir=$(cd $(dirname $0);pwd)
dir=$(dirname $script_dir)

# /home/bric/Workspace/context_sm/cuda/src/memory
echo "$script_dir"

# /home/bric/Workspace/context_sm/cuda/src
echo "$dir"

# /home/bric/Workspace/context_sm/cuda
echo $(dirname $dir)

echo $#
