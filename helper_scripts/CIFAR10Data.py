import lightning.pytorch as pl
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
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def prepare_data(self):
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])
        if stage == 'fit' or stage is None:
            self.cifar10_train = CIFAR10(
                root=self.data_dir, train=True, download=False, transform=self.train_transform or transform
            )
            self.cifar10_val = CIFAR10(
                root=self.data_dir, train=False, download=False, transform=self.val_transform or transform
            )
        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(
                root=self.data_dir, train=False, download=False, transform=self.val_transform or transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar10_val,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar10_test,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )