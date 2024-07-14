from torch import nn, optim
import lightning.pytorch as pl
import timm
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets

from helper_scripts.utils import load_dataset


class CIFAR10Module(pl.LightningModule):
    def __init__(self, model_name, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=10)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.acc = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.acc(outputs, labels))
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.acc(outputs, labels))
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("test_loss", loss)
        self.log("test_acc", self.acc(outputs, labels))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def create_dataloader(self, train = True):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=train, num_workers=4)
        return loader

    def train_dataloader(self):
        return load_dataset(batch_size=32, num_workers=4, pin_memory=True, train=True)

    def val_dataloader(self):
        return load_dataset(batch_size=32, num_workers=4, pin_memory=True, train=False)

    def test_dataloader(self):
        return load_dataset(batch_size=32, num_workers=4, pin_memory=True, train=False)

    def children(self, *args, **kwargs):
        return self.model.children(*args, **kwargs)
