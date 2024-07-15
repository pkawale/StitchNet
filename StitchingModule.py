from torch import nn, optim
import pytorch_lightning as pl
from stitching_layer import StitchingModel

class LightningStitchingModule(pl.LightningModule):
    # QUESTION: is this class used anywhere? I see stitching_layer.StitchingModel and StitchingModelModule.StitchingModelModule being used
    # (which are already redundant, perhaps), but this one is not used?
    def __init__(self, model1, model2, split1, split2):
        super(LightningStitchingModule, self).__init__()
        self.stitching_model = StitchingModel(model1, model2, split1, split2)
        self.criterion = nn.CrossEntropyLoss()

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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.stitching_model.parameters(), lr=1e-3)
        return optimizer
