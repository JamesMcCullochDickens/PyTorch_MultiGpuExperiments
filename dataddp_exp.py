import torch
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
    setup(rank, world_size)
    model = resnet50(pretrained=False).to(rank)
    model = DDP(model, device_ids=[rank])
    batch_sizes = [4, 8, 16, 20]
    num_epochs = 5
    for batch_size in batch_sizes:
        dl = DataLoader(dataset=RandomImagesDataloader(),
                        batch_size=batch_size, shuffle=True,
                        num_workers=1, drop_last=True)
        optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        t1 = time.time()
        for epoch_num in range(num_epochs):
            for batch_num, batch in enumerate(dl):
                targets = torch.randint(size=(batch_size,), low=0, high=1000).long().to(rank)
                batch = batch.to(rank)
                output = model(batch)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
        # dealing with synchronization issues
        time.sleep(1.0)
        t2 = time.time()
        if rank == 0:
            print(f"The estimated training time for {world_size} gpu/s at batch size "
                  f"{batch_size} is {round(t2-t1, 3)} seconds")
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    spawn_processes(train, world_size)
