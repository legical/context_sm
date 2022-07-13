# context_sm 环境变量
输入env检查
```
export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```
# 使用
## 编译
```
nvcc -arch sm_86 -lcuda -o test kernel.cu util.cu
```

## 开启MPS （root）
```
chmod +x ./choose_as_root.bash
```
输入y

## 运行
```
./test -k 4 -s 4 4 6 2 -b 8
```

其中
* -k kernelnum 并行运行kernelnum个kernel
* -s sm_num0 sm_num1 ... sm_num_kernelnum-1 分别为每个kernel绑定对应数量的sm，如果没指定数量，则默认绑定2个sm
* -b block_num 每个kernel启动多少个block,每个kernel启动的block数量一致

## see all GPUs

```
nvidia-smi -L
```

## Starting MPS control daemon

As root, run the commands

Select GPU 0
```
export CUDA_VISIBLE_DEVICES=0
```
Set GPU 0 to exclusive mode
```
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
```
Start the daemon
```
nvidia-cuda-mps-control -d
```

This will start the MPS control daemon that will spawn a new MPS Server instance for any $UID starting an application and associate it with the GPU visible to the control daemon. Note that CUDA_VISIBLE_DEVICES should not be set in the client process’s environment.

## comfiled order (RTX30 series)
```
nvcc -arch sm_86 -lcuda -o test testone.cu util.cu
```

## Shutting Down MPS control daemon

To shut down the daemon, as root, run

```
echo quit | nvidia-cuda-mps-control
```

## more order about mps

### nvidia-cuda-mps-control

#### Describes usage of this utility.

```
man nvidia-cuda-mps-control
```

#### Start daemon in background process.

```
nvidia-cuda-mps-control -d 
```

#### See if the MPS daemon is running.

```
ps -ef | grep mps
```

#### Shut the daemon down.

```
echo quit | nvidia-cuda-mps-control
```

#### Start deamon in foreground

```
nvidia-cuda-mps-control -f
```

## -arch=sm_xx

```
Pascal(CUDA 8 and later)：
SM60 or SM_60, compute_60 – GP100/Tesla P100 – DGX-1 (Generic Pascal)

SM61 or SM_61, compute_61 – GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2

SM62 or SM_62, compute_62 – Integrated GPU on the NVIDIA Drive PX2, Tegra (Jetson) TX2

Volta(CUDA 9 and later)：
SM70 or SM_70, compute_70 – DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100

SM72 or SM_72, compute_72 – Jetson AGX Xavier, Drive AGX Pegasus, Xavier NX

Turing(CUDA 10 and later)：
SM75 or SM_75, compute_75 – GTX/RTX Turing – GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, Titan RTX, Quadro RTX 4000, Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000, Quadro T1000/T2000, Tesla T4

Ampere(CUDA 11 and later)：
SM80 or SM_80, compute_80 – NVIDIA A100 (the name “Tesla” has been dropped – GA100), NVIDIA DGX-A100

SM86 or SM_86, compute_86 – (from CUDA 11.1 onwards) Tesla GA10x cards, RTX Ampere – RTX 3080, GA102 – RTX 3090, RTX A6000, RTX A40, GA106 – RTX 3060, GA104 – RTX 3070, GA107 – RTX 3050
```

When used in interactive mode, the available commands are

* get_server_list – this will print out a list of all PIDs of server instances.
* start_server –uid `<user id>` - this will manually start a new instance of nvidia-cuda-mps-server with the given user ID.
* get_client_list `<PID>` - this lists the PIDs of client applications connected to a server instance assigned to the given PID
* quit – terminates the nvidia-cuda-mps-control daemon
* Commands available to Volta MPS control:
* get_device_client_list [`<PID>`] - this lists the devices and PIDs of client applications that enumerated this device. It optionally takes the server instance PID.
* set_default_active_thread_percentage `<percentage>` - this overrides the default active thread percentage for MPS servers. If there is already a server spawned, this command will only affect the next server. The set value is lost if a quit command is executed. The default is 100.
* get_default_active_thread_percentage - queries the current default available thread percentage.
* set_active_thread_percentage `<PID>` `<percentage>` - this overrides the active thread percentage for the MPS server instance of the given PID. All clients created with that server afterwards will observe the new limit. Existing clients are not affected.
* get_active_thread_percentage `<PID>` - queries the current available thread percentage of the MPS server instance of the given PID.
* set_default_device_pinned_mem_limit `<dev>` `<value>` - this sets the default device pinned memory limit for each MPS client. If there is already a server spawned, this command will only affect the next server. The set value is lost if a quit command is executed. The value must be in the form of an integer followed by a qualifier, either “G” or “M” that specifies the value in Gigabyte or Megabyte respectively. For example: In order to set limit to 10 gigabytes for device 0, the command used is: set_default_device_pinned_mem_limit 0 10G. By default memory limiting is disabled.
* get_default_device_pinned_mem_limit `<dev>` - queries the current default pinned memory limit for the device.
* set_device_pinned_mem_limit `<PID>` `<dev>` `<value>` - this overrides the device pinned memory limit for MPS servers. This sets the device pinned memory limit for each client of MPS server instance of the given PID for the device dev. All clients created with that server afterwards will observe the new limit. Existing clients are not affected. Example usage to set memory limit of 900MB for server with pid 1024 for device 0.set_device_pinned_mem_limit 1024 0 900M
* get_device_pinned_mem_limit `<PID>` `<dev>` - queries the current device pinned memory limit of the MPS server instance of the given PID for the device dev.

Only one instance of the nvidia-cuda-mps-control daemon should be run per node.
