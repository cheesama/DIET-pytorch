from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

# related to pretrained model
from transformers import ElectraModel, ElectraTokenizer
from kobert_transformers import get_kobert_model, get_distilkobert_model
from kobert_transformers import get_tokenizer as kobert_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingTransformer(nn.Module):
    def __init__(
        self,
        backbone: None,
        vocab_size: int,
        seq_len: int,
        intent_class_num: int,
        entity_class_num: int,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        pad_token_id: int = 0,
    ):
        super(EmbeddingTransformer, self).__init__()
        self.backbone = backbone
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

        if backbone is None:
            self.encoder = nn.TransformerEncoder(
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation
                ),
                num_encoder_layers,
                LayerNorm(d_model),
            )
        else:  # pre-defined model architecture use
            if backbone == "kobert":
                self.encoder = get_kobert_model()
            elif backbone == "distill_kobert":
                self.encoder = get_distilkobert_model()
            elif backbone == "koelectra":
                self.encoder = ElectraModel.from_pretrained(
                    "monologg/koelectra-small-v2-discriminator"
                )

            d_model = self.encoder.config.hidden_size

        self.embedding = nn.EmbeddingBag(vocab_size, d_model, sparse=True)
        self.position_embedding = nn.EmbeddingBag(self.seq_len, d_model, sparse=True)

        self.intent_feature = nn.Linear(d_model, intent_class_num)
        self.entity_feature = nn.Linear(d_model, entity_class_num)

    def forward(self, x):
        if self.backbone is None:
            src_key_padding_mask = x == self.pad_token_id
            embedding = self.embedding(x)
            embedding += self.position_embedding(
                torch.arange(self.seq_len).repeat(x.size(0), 1).type_as(x)
            )

            feature = self.encoder(
                embedding.transpose(1, 0), src_key_padding_mask=src_key_padding_mask
            )  # (N,S,E) -> (S,N,E)

            # first token in sequence used to intent classification
            intent_feature = self.intent_feature(feature[0, :, :])  # (N,E) -> (N,i_C)

            # other tokens in sequence used to entity classification
            entity_feature = self.entity_feature(
                feature[:, :, :]
            )  # (S,N,E) -> (S,N,e_C)

            return intent_feature, entity_feature.transpose(1, 0)[:, :, :]

        elif self.backbone in ["kobert", "distill_kobert", "koelectra"]:
            feature = self.encoder(x)

            if type(feature) == tuple:
                feature = feature[0]  # last_hidden_state (N,S,E)

            # first token in sequence used to intent classification
            intent_feature = self.intent_feature(feature[:, 0, :]) # (N,E) -> (N,i_C)

            # other tokens in sequence used to entity classification
            entity_feature = self.entity_feature(feature[:, :, :]) # (N,S,E) -> (N,S,e_C)

            return intent_feature, entity_feature
