from pytorch_lightning import Trainer

from argparse import Namespace

from .DIET_lightning_model import DualIntentEntityTransformer

import os, sys
import torch


def train(
    file_path,
    #training args
    train_ratio=0.8,
    batch_size=32,
    optimizer="Adam",
    lr=5e-5,
    checkpoint_path=os.getcwd(),
    max_epochs=10,
    #model args
    num_encoder_layers=1,
    **kwargs
):
    gpu_num = torch.cuda.device_count()

    trainer = Trainer(
        default_root_dir=checkpoint_path, max_epochs=max_epochs, gpus=gpu_num
    )

    model_args = {}

    #training args
    model_args["nlu_data"] = open(file_path, encoding='utf-8').readlines()
    model_args["train_ratio"] = train_ratio
    model_args["batch_size"] = batch_size
    model_args["optimizer"] = optimizer
    model_args["lr"] = lr
    
    #model args
    model_args["num_encoder_layers"] = num_encoder_layers

    for key, value in kwargs.items():
        model_args[key] = value

    hparams = Namespace(**model_args)

    model = DualIntentEntityTransformer(hparams)

    trainer.fit(model)
