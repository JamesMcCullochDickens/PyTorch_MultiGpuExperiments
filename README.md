# PyTorch_MultiGpuExperiments
A repo to perform benchmarks and experiments for my local multi-gpu setup.
My current rig has 2 GPUs: 3090ti and a 3090 (the 3090 is, unfortunately, in a PCI Gen 3.0
4x functionality with a 16x sized slot).
My CPU is an intel i9-11900K (8 cores).

I'm using PyTorch version 1.11.0. For Windows I am using the gloo backend for 
the data distributed parallel experiments, and for Linux (tested on Ubuntu), I
am using nccl. 

The benchmark consists of declaring a dataset of random images fed into a ResNet50
for various batch sizes for 5 epochs.


