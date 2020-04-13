from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import Namespace

from .DIET_lightning_model import DualIntentEntityTransformer

import os, sys
import torch


def train(
    file_path,
    train_ratio=0.8,
    batch_size=32,
    optimizer="Adam",
    lr=1e-4,
    checkpoint_path=os.getcwd(),
    max_epochs=10,
    **kwargs
):
    gpu_num = torch.cuda.device_count()

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path
        + os.sep
        + "DIET_{epoch:02d}-{val_loss:.2f}-{intent_acc:.3f}-{entity_acc:.3f}"
    )

    if gpu_num > 0:
        trainer = Trainer(
            default_root_dir=checkpoint_path, max_epochs=max_epochs, gpus=gpu_num
        )
    else:
        trainer = Trainer(default_save_path=checkpoint_path, max_epochs=max_epochs)

    trainer.checkpoint_callback = checkpoint_callback

    model_args = {}
    model_args["data_file_path"] = file_path
    model_args["train_ratio"] = train_ratio
    model_args["batch_size"] = batch_size
    model_args["optimizer"] = optimizer
    model_args["lr"] = lr

    for key, value in kwargs.items():
        model_args[key] = value

    hparams = Namespace(**model_args)

    model = DualIntentEntityTransformer(hparams)

    trainer.fit(model)
