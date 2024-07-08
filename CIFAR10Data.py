import lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10


class CIFAR10Data(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        train_transform=None,
        val_transform=None,
        num_workers=4,
        pin_memory=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        transform = (
            self.train_transform
            if self.train_transform
            else T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            )
        )
        dataset = CIFAR10(
            root=self.data_dir, train=True, download=False, transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = (
            self.val_transform
            if self.val_transform
            else T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            )
        )
        dataset = CIFAR10(
            root=self.data_dir, train=False, download=False, transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
