from pytorch_lightning import Trainer
from transformers import ElectraTokenizer
from argparse import Namespace

from .DIET_lightning_model import DualIntentEntityTransformer

import os, sys
import torch


def train(
    file_path,
    # training args
    train_ratio=0.8,
    batch_size=32,
    optimizer="Adam",
    intent_optimizer_lr=1e-5,
    entity_optimizer_lr=2e-5,
    checkpoint_path=os.getcwd(),
    max_epochs=20,
    tokenizer=ElectraTokenizer.from_pretrained(
        "monologg/koelectra-small-discriminator"
    ),
    # tokenizer=None,
    # model args
    # refer to https://www.notion.so/A-Primer-in-BERTology-What-we-know-about-how-BERT-works-aca45feaba2747f09f1a3cdd1b1bbe16
    d_model=256,
    num_encoder_layers=2,
    **kwargs
):
    gpu_num = min(1, torch.cuda.device_count())

    """
    if gpu_num > 1:
        trainer = Trainer(
            default_root_dir=checkpoint_path,
            max_epochs=max_epochs,
            gpus=gpu_num,
            distributed_backend="dp",
        )
    else:
    """

    trainer = Trainer(
        default_root_dir=checkpoint_path, max_epochs=max_epochs, gpus=gpu_num
    )

    model_args = {}

    # training args
    model_args["max_epochs"] = max_epochs
    model_args["nlu_data"] = open(file_path, encoding="utf-8").readlines()
    model_args["train_ratio"] = train_ratio
    model_args["batch_size"] = batch_size
    model_args["optimizer"] = optimizer
    model_args["intent_optimizer_lr"] = intent_optimizer_lr
    model_args["entity_optimizer_lr"] = entity_optimizer_lr

    if type(tokenizer) == ElectraTokenizer:
        model_args["tokenizer"] = tokenizer

    # model args
    model_args["d_model"] = d_model
    model_args["num_encoder_layers"] = num_encoder_layers

    for key, value in kwargs.items():
        model_args[key] = value

    hparams = Namespace(**model_args)

    model = DualIntentEntityTransformer(hparams)

    trainer.fit(model)
