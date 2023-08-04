import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from monai.losses import DiceLoss

from ai.utils import get_accuracy

class SegmentationModule(pl.LightningModule):
    def __init__(self, model: nn.Module, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = DiceLoss(sigmoid=True)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        y_pred_label = (y_hat >= 0.5).type(torch.float32)*255
        accuracy = get_accuracy(y, y_pred_label)

        self.log("train_acc", accuracy, on_epoch=True, on_step=False)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        y_pred_label = (y_hat >= 0.5).type(torch.float32)*255
        accuracy = get_accuracy(y, y_pred_label)

        self.log("val_acc", accuracy, on_epoch=True, on_step=False)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
