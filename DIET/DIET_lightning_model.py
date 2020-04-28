from argparse import Namespace

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from torchnlp.metrics import get_accuracy, get_token_accuracy

from pytorch_lightning import Trainer

from .dataset.intent_entity_dataset import RasaIntentEntityDataset
from .model.models import EmbeddingTransformer

import os, sys
import multiprocessing
import dill

import torch
import torch.nn as nn
import pytorch_lightning as pl


class DualIntentEntityTransformer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.dataset = RasaIntentEntityDataset(markdown_lines=self.hparams.nlu_data, tokenize_fn=self.hparams.tokenize_fn)

        self.model = EmbeddingTransformer(
            vocab_size=self.dataset.get_vocab_size(),
            seq_len=self.dataset.get_seq_len(),
            intent_class_num=len(self.dataset.get_intent_idx()),
            entity_class_num=len(self.dataset.get_entity_idx()),
            num_encoder_layers=self.hparams.num_encoder_layers,
        )

        self.train_ratio = self.hparams.train_ratio
        self.batch_size = self.hparams.batch_size
        self.optimizer = self.hparams.optimizer
        self.intent_optimizer_lr = self.hparams.intent_optimizer_lr
        self.entity_optimizer_lr = self.hparams.entity_optimizer_lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        train_length = int(len(self.dataset) * self.train_ratio)

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_length, len(self.dataset) - train_length],
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )
        return val_loader

    def configure_optimizers(self):
        intent_optimizer = eval(
            f"{self.optimizer}(self.parameters(), lr={self.intent_optimizer_lr})"
        )
        entity_optimizer = eval(
            f"{self.optimizer}(self.parameters(), lr={self.entity_optimizer_lr})"
        )

        return (
            [intent_optimizer, entity_optimizer],
            # [StepLR(intent_optimizer, step_size=1),StepLR(entity_optimizer, step_size=1),],
            [
                ReduceLROnPlateau(intent_optimizer, patience=1),
                ReduceLROnPlateau(entity_optimizer, patience=1),
            ],
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.model.train()

        tokens, intent_idx, entity_idx = batch

        intent_pred, entity_pred = self.forward(tokens)

        intent_acc = get_accuracy(intent_idx.cpu(), intent_pred.max(1)[1].cpu())[0]
        entity_acc = get_token_accuracy(
            entity_idx.cpu(),
            entity_pred.max(2)[1].cpu(),
            ignore_index=self.dataset.pad_token_id,
        )[0]

        tensorboard_logs = {
            "train/intent/acc": intent_acc,
            "train/entity/acc": entity_acc,
        }

        if optimizer_idx == 0:
            intent_loss = self.loss_fn(intent_pred, intent_idx.squeeze(1))
            tensorboard_logs["train/intent/loss"] = intent_loss
            return {
                "loss": intent_loss,
                "log": tensorboard_logs,
            }
        if optimizer_idx == 1:
            entity_loss = self.loss_fn(entity_pred.transpose(1, 2), entity_idx.long())
            tensorboard_logs["train/entity/loss"] = entity_loss
            return {
                "loss": entity_loss,
                "log": tensorboard_logs,
            }

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        tokens, intent_idx, entity_idx = batch

        intent_pred, entity_pred = self.forward(tokens)

        intent_acc = get_accuracy(intent_idx.cpu(), intent_pred.max(1)[1].cpu())[0]
        entity_acc = get_token_accuracy(
            entity_idx.cpu(),
            entity_pred.max(2)[1].cpu(),
            ignore_index=self.dataset.pad_token_id,
        )[0]

        intent_loss = self.loss_fn(intent_pred, intent_idx.squeeze(1))
        entity_loss = self.loss_fn(
            entity_pred.transpose(1, 2), entity_idx.long()
        )  # , ignore_index=0)

        return {
            "val_intent_acc": torch.Tensor([intent_acc]),
            "val_entity_acc": torch.Tensor([entity_acc]),
            "val_intent_loss": intent_loss,
            "val_entity_loss": entity_loss,
            "val_loss": intent_loss + entity_loss,
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_intent_acc = torch.stack([x["val_intent_acc"] for x in outputs]).mean()
        avg_entity_acc = torch.stack([x["val_entity_acc"] for x in outputs]).mean()

        tensorboard_logs = {
            "val/loss": avg_loss,
            "val/intent_acc": avg_intent_acc,
            "val/entity_acc": avg_entity_acc,
        }

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
