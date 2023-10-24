from pprint import pprint

import torch
import cpuinfo

cpu_info = cpuinfo.get_cpu_info()
vendor = cpu_info['vendor_id_raw']
model = cpu_info['brand_raw']
num_cores = cpu_info['count']
architecture = cpu_info['arch']
frequency = cpu_info['hz_actual_friendly']
bits = cpu_info['bits']

# Get the memory size in GiB
with open('/proc/meminfo') as f:
    meminfo = f.read().splitlines()
    meminfo = [x for x in meminfo if 'MemTotal' in x][0]
    meminfo = meminfo.split()
    gibibyte = round(int(meminfo[1]) / 1024**2, 1)
    gigabyte = round(int(meminfo[1]) / 1000**2, 1)

print(f"CPU: {vendor} {model}, {num_cores} cores, {architecture} {bits}-bit architecture @ {frequency}, {gibibyte} GiB ({gigabyte} GB)")

if not torch.cuda.is_available():
    print("CUDA is not supported")
    exit(0)

num_gpus = torch.cuda.device_count()
print(f"CUDA is supported, found {num_gpus} GPU" + ("s" if num_gpus > 1 else ""))

for current_gpu in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(current_gpu)
    gpu_capability = ".".join([str(x) for x in torch.cuda.get_device_capability(current_gpu)])
    gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory
    gibibyte = round(gpu_memory / 1024**3, 1)
    gigabyte = round(gpu_memory / 1000**3, 1)
    print(f"GPU {current_gpu}: {gpu_name} {gpu_capability}, {gibibyte} GiB ({gigabyte} GB)")

