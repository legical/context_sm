# context_sm

```
export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```

## see all GPUs

```
nvidia-smi -L
```

## Starting MPS control daemon

As root, run the commands

```
export CUDA_VISIBLE_DEVICES=0 # Select GPU 0.
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS # Set GPU 0 to exclusive mode.
nvidia-cuda-mps-control -d # Start the daemon.
```

This will start the MPS control daemon that will spawn a new MPS Server instance for any $UID starting an application and associate it with the GPU visible to the control daemon. Note that CUDA_VISIBLE_DEVICES should not be set in the client process’s environment.

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
GeForce RTX 3070, 3080, 3090
ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]

Kepler GeForce GTX 770, GTX 760, GT 740
ARCH= -gencode arch=compute_30,code=sm_30

Tesla A100 (GA100), DGX-A100, RTX 3080
ARCH= -gencode arch=compute_80,code=[sm_80,compute_80]

Tesla V100
ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]

GeForce RTX 2080 Ti, RTX 2080, RTX 2070, Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000, Tesla T4, XNOR Tensor Cores
ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]

Jetson XAVIER
ARCH= -gencode arch=compute_72,code=[sm_72,compute_72]

GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

GP100/Tesla P100 - DGX-1
ARCH= -gencode arch=compute_60,code=sm_60

For Jetson TX1, Tegra X1, DRIVE CX, DRIVE PX - uncomment:
ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

For Jetson Tx2 or Drive-PX2 uncomment:
ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]

For Tesla GA10x cards, RTX 3090, RTX 3080, RTX 3070, RTX A6000, RTX A40 uncomment:
ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]
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
