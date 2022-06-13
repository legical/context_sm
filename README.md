# context_sm

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

When used in interactive mode, the available commands are

* get_server_list – this will print out a list of all PIDs of server instances.
* start_server –uid <user id> - this will manually start a new instance of nvidia-cuda-mps-server with the given user ID.
* get_client_list <PID> - this lists the PIDs of client applications connected to a server instance assigned to the given PID
* quit – terminates the nvidia-cuda-mps-control daemon
* Commands available to Volta MPS control:
* get_device_client_list [<PID>] - this lists the devices and PIDs of client applications that enumerated this device. It optionally takes the server instance PID.
* set_default_active_thread_percentage <percentage> - this overrides the default active thread percentage for MPS servers. If there is already a server spawned, this command will only affect the next server. The set value is lost if a quit command is executed. The default is 100.
* get_default_active_thread_percentage - queries the current default available thread percentage.
* set_active_thread_percentage <PID> <percentage> - this overrides the active thread percentage for the MPS server instance of the given PID. All clients created with that server afterwards will observe the new limit. Existing clients are not affected.
* get_active_thread_percentage <PID> - queries the current available thread percentage of the MPS server instance of the given PID.
* set_default_device_pinned_mem_limit <dev> <value> - this sets the default device pinned memory limit for each MPS client. If there is already a server spawned, this command will only affect the next server. The set value is lost if a quit command is executed. The value must be in the form of an integer followed by a qualifier, either “G” or “M” that specifies the value in Gigabyte or Megabyte respectively. For example: In order to set limit to 10 gigabytes for device 0, the command used is: set_default_device_pinned_mem_limit 0 10G. By default memory limiting is disabled.
* get_default_device_pinned_mem_limit <dev> - queries the current default pinned memory limit for the device.
* set_device_pinned_mem_limit <PID> <dev> <value> - this overrides the device pinned memory limit for MPS servers. This sets the device pinned memory limit for each client of MPS server instance of the given PID for the device dev. All clients created with that server afterwards will observe the new limit. Existing clients are not affected. Example usage to set memory limit of 900MB for server with pid 1024 for device 0.set_device_pinned_mem_limit 1024 0 900M
* get_device_pinned_mem_limit <PID> <dev> - queries the current device pinned memory limit of the MPS server instance of the given PID for the device dev.
  
Only one instance of the nvidia-cuda-mps-control daemon should be run per node.
