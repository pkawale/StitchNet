from torch import nn, optim
import lightning.pytorch as pl
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MulticlassAccuracy
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
        self.acc = MulticlassAccuracy(num_classes=10).to(self.device)

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
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self(x)  # Ensure y_hat is defined as the model's output
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)

        # Move metric calculation to the correct device
        self.acc = self.acc.to(self.device)
        self.log("test_acc", self.acc(y_hat, y), prog_bar=True)
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

    def children(self, *args, **kwargs):
        return self.model.children(*args, **kwargs)
