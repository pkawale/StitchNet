from torch import nn, optim
from lightning import LightningModule
import timm
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CIFAR10Module(LightningModule):
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
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
