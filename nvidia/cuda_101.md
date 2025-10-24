# cuda 101

CUDA 全称 Compute _Unified Devices_ Architecture 是使用大量 GPU 计算单元进行通用计算的平台。

CUDA C/C++ 是基于标准 C/C++ 的拓展，实现异构计算。
这里说的异构计算主要指
* 主机 cpu host 和主机内存 host memory
* GPU 设备 device 和设备内存 device memory

CPU 架构有复杂的缓存体系，指令延迟非常小，大量的晶体管处理逻辑任务；
GPU 拥有更多的计算单元和高带宽，有更高的吞吐，大量的晶体管处理数值和逻辑运算；



