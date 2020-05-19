from pytorch_lightning import Trainer
from transformers import ElectraTokenizer
from argparse import Namespace

from .DIET_lightning_model import DualIntentEntityTransformer
from .dataset.intent_entity_dataset import RasaIntentEntityDataset

import os, sys
import torch
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.base import Callback
from DIET.metrics import show_intent_report


class PerfCallback(Callback):
    def __init__(self, file_path=None, gpu_num=0):
        self.file_path = file_path
        if gpu_num > 0:
            self.cuda = True
        else:
            self.cuda = False

    def on_train_end(self, trainer, pl_module):
        print("train finished")
        if self.file_path is None:
            dataset = pl_module.val_dataset
        else:
            nlu_data = open(self.file_path, encoding="utf-8").readlines()
            dataset = RasaIntentEntityDataset(nlu_data, tokenizer=pl_module.hparams.tokenizer)
        
        dataloader = DataLoader(dataset, batch_size = 32)
        
        show_intent_report(dataset, pl_module, file_name="test_metric.json", output_dir="results", cuda=True)


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
    tokenizer=ElectraTokenizer.from_pretrained("monologg/koelectra-small-discriminator"),
    #tokenizer=None,
    # model args
    num_encoder_layers=1,
    **kwargs
):
    gpu_num = torch.cuda.device_count()
    
    trainer = Trainer(
        default_root_dir=checkpoint_path, max_epochs=max_epochs, gpus=gpu_num, callbacks=[PerfCallback(file_path = file_path, gpu_num=gpu_num)]
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
    model_args["num_encoder_layers"] = num_encoder_layers

    for key, value in kwargs.items():
        model_args[key] = value

    hparams = Namespace(**model_args)

    model = DualIntentEntityTransformer(hparams)

    trainer.fit(model)
