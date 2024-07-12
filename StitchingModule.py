import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import pytorch_lightning as pl

from stitching_layer import StitchingModel


class LightningStitchingModel(pl.LightningModule):
    def __init__(self, model1, model2, split1, split2):
        super(LightningStitchingModel, self).__init__()
        self.stitching_model = StitchingModel(model1, model2, split1, split2)
        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        return self.stitching_model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)
        return loss
        # _, predicted = torch.max(outputs.data, 1)
        # accuracy = (predicted == labels).sum().item() / labels.size(0)
        # self.log('val_loss', loss, prog_bar=True)
        # self.log('val_accuracy', accuracy, prog_bar=True)
        # return {"val_loss": loss, "val_accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = Adam(self.stitching_model.parameters(), lr=1e-3)
        return optimizer
