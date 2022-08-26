# PyTorch_MultiGpuExperiments
A repo to perform benchmarks and experiments for my local multi-gpu setup.
My current rig has 2 GPUs: 3090ti and a 3090 (the 3090 is, unfortunately, in a PCI Gen 3.0
4x functionality with a 16x sized slot).
My CPU is an intel i9-11900K (8 cores).

I'm using PyTorch version 1.12.0. For Windows I am using the gloo backend for 
the data distributed parallel experiments, and for Linux (tested on Ubuntu), I
am using nccl (there are some issues with nccl hanging, will investigate more later).

The benchmark consists of declaring a dataset of random images fed into a ResNet50
for various batch sizes for 5 epochs, 1000 random images of size 600 * 600.

### Ubuntu 22.04 LTS. 

## Results for Single GPU training

The estimated training time for 1 gpu/s at batch size 8 is 87.884 seconds

The estimated training time for 1 gpu/s at batch size 16 is 86.74 seconds


## Results for Data Parallel training with 2 gpus

The estimated training time for 2 gpu/s at batch size 8 is 108.168 seconds

The estimated training time for 2 gpu/s at batch size 16 is 75.209 seconds


## Results for Distributed Data Parallel training with 2 gpus

The estimated training time for 2 gpu/s at batch size 8 is 50.848 seconds

The estimated training time for 2 gpu/s at batch size 16 is 47.357 seconds

### Windows 10

## Results for Single GPU training

The estimated training time for 1 gpu/s at batch size 8 is 91.801 seconds

The estimated training time for 1 gpu/s at batch size 16 is 87.89 seconds


## Results for Data Parallel training with 2 gpus

The estimated training time for 2 gpu/s at batch size 8 is 106.887 seconds

The estimated training time for 2 gpu/s at batch size 16 is 77.81 seconds


## Results for Distributed Data Parallel training with 2 gpus

The estimated training time for 2 gpu/s at batch size 8 is 65.234 seconds

The estimated training time for 2 gpu/s at batch size 16 is 52.446 seconds
