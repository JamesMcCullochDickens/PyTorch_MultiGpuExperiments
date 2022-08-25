import os
import torch

# set all gpus as visible
n_gpus = torch.cuda.device_count()
visible_devices = ""
for i in range(n_gpus):
    if i != n_gpus-1:
        visible_devices += str(i)+","
    else:
        visible_devices += str(i)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DECIVES"] = visible_devices

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.models import resnet50
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist
from sys import platform
import torch.multiprocessing as mp


def is_windows():
    return platform == "win32"


def spawn_processes(fn, world_size):
    mp.spawn(fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo" if is_windows() else "nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class RandomImagesDataloader(Dataset):
    def __init__(self, num_images=500, height=600, width=600, num_channels=3):
        self.num_images = num_images
        self.dataset = torch.randn(num_images, num_channels, height, width)
        self.len = num_images

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.len


def train(rank, world_size):
    # declare the model
    setup(rank, world_size)
    model = resnet50(pretrained=False).to(rank)
    model = DDP(model, device_ids=[rank])

    # declare the training params
    batch_sizes = [2, 4, 8, 16, 32]
    num_epochs = 5

    for batch_size in batch_sizes:
        # declare the dataloader
        dl = DataLoader(dataset=RandomImagesDataloader(),
                        batch_size=batch_size, shuffle=True,
                        num_workers=1, drop_last=True)

        # declare the optimizer and loss function
        optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        t1 = time.time()
        # the training loop
        for epoch_num in range(num_epochs):
            for batch_num, batch in enumerate(dl):
                # load targets to the gpu, compute the loss and backpropagate
                targets = torch.randint(size=(batch_size,), low=0, high=1000).long().to(rank)
                batch = batch.cuda()
                output = model(batch)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()

        # dealing with synchronization issues
        time.sleep(4.0)
        t2 = time.time()

        if rank == 0:
            print(f"The estimated training time for {world_size} gpu/s at batch size {batch_size} is {round(t2-t1, 3)} seconds")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    spawn_processes(train, world_size)

