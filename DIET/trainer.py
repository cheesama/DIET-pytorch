from pytorch_lightning import Trainer
from argparse import Namespace

from torchnlp.encoders.text import CharacterEncoder, WhitespaceEncoder

# related to pretrained tokenizer & model
from transformers import ElectraModel, ElectraTokenizer
from kobert_transformers import get_kobert_model, get_distilkobert_model
from kobert_transformers import get_tokenizer as kobert_tokenizer

from .DIET_lightning_model import DualIntentEntityTransformer

import os, sys
import torch


def train(
    file_path,
    # training args
    train_ratio=0.8,
    batch_size=128,
    optimizer="Adam",
    intent_optimizer_lr=1e-5,
    entity_optimizer_lr=2e-5,
    checkpoint_path=os.getcwd(),
    max_epochs=15,
    tokenizer_type="char",
    # model args
    # refered below link to optimize model
    # https://www.notion.so/A-Primer-in-BERTology-What-we-know-about-how-BERT-works-aca45feaba2747f09f1a3cdd1b1bbe16
    backbone=None,
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

    if backbone is not None:
        if backbone not in ["kobert", "distill_kobert", "koelectra"]:
            assert ValueError(
                "backbone shoulud be None or kobert or distill_kobert or koelectra"
            )

    if tokenizer_type not in ["char", "space"]:
        assert ValueError("tokenizer_type should be char or space")

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

    if backbone is None:
        if tokenizer_type == "char":
            model_args["tokenizer"] = CharacterEncoder
        elif tokenizer_type == "space":
            model_args["tokenizer"] = WhitespaceEncoder

        # torchnlp special token indices
        model_args["tokenizer"].cls_token_id = 3
        model_args["tokenizer"].sep_token_id = 2

    else:
        if backbone in ["kobert", "distill_kobert"]:
            model_args["tokenizer"] = kobert_tokenizer()
        elif backbone == "koelectra":
            model_args["tokenizer"] = ElectraTokenizer.from_pretrained(
                "monologg/koelectra-small-discriminator"
            )

    # model args
    model_args["backbone"] = backbone
    model_args["d_model"] = d_model
    model_args["num_encoder_layers"] = num_encoder_layers

    for key, value in kwargs.items():
        model_args[key] = value

    hparams = Namespace(**model_args)

    model = DualIntentEntityTransformer(hparams)

    trainer.fit(model)
