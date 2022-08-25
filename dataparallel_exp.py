import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.models import resnet50
import time


class RandomImagesDataloader(Dataset):
    def __init__(self, num_images=500, height=600, width=600, num_channels=3):
        self.num_images = num_images
        self.dataset = torch.randn(num_images, num_channels, height, width)
        self.len = num_images

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.len


def sample_train(n_gpus):
    model = resnet50(pretrained=False)
    # use data parallelism if there is more than one gpu
    if n_gpus > 1:
        device_ids = list(range(n_gpus))
        model = nn.DataParallel(model, device_ids).cuda()
    else:
        model = model.cuda()
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
                targets = torch.randint(size=(batch_size,), low=0, high=1000).long().cuda()
                batch = batch.cuda()
                output = model(batch)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
        # dealing with synchronization issues
        time.sleep(1.0)
        t2 = time.time()
        print(f"The estimated training time for {num_gpus} gpu/s at batch size "
              f"{batch_size} is {round(t2-t1, 3)} seconds")


if __name__ == "__main__":
    for num_gpus in range(1, 3):
        print(f"Training with {num_gpus} gpu/s")
        sample_train(n_gpus=num_gpus)
        print("\n")
