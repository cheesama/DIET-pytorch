from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

import os, sys
import torch
import pytorch_lightning as pl

class DualIntentEntityTransformer(pl.LightningModule):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def prepare_data(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def valdition_epoch_end(self, outputs):
        pass