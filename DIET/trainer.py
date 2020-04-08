from pytorch_lightning import Trainer
from argparse import Namespace

from .DIET_lightning_model import DualIntentEntityTransformer

import os, sys


def train(
    file_path,
    train_ratio=0.8,
    batch_size=32,
    optimizer="Adam",
    lr=1e-3,
    checkpoint_path=os.getcwd(),
    max_epochs=10,
    **kwargs
):
    trainer = Trainer(default_save_path=checkpoint_path, max_epochs=max_epochs,)

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
