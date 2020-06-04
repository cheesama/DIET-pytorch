from torchnlp.encoders.text import CharacterEncoder, WhitespaceEncoder

from .DIET_lightning_model import DualIntentEntityTransformer

import torch
import torch.nn as nn

import logging

model = None
intent_dict = {}
entity_dict = {}

class Inferencer:
    def __init__(self, checkpoint_path: str):
        self.model = DualIntentEntityTransformer.load_from_checkpoint(checkpoint_path)
        self.model.model.eval()

        self.intent_dict = {}
        for k, v in self.model.dataset.intent_dict.items():
            self.intent_dict[v] = k # str key -> int key

        self.entity_dict = {}
        for k, v in self.model.dataset.entity_dict.items():
            self.entity_dict[v] = k # str key -> int key

        logging.info("intent dictionary")
        logging.info(self.intent_dict)
        print()

        logging.info("entity dictionary")
        logging.info(self.entity_dict)

    def is_same_entity(i, j):
        # check whether XXX_B, XXX_I tag are same 
        return self.entity_dict[i][:self.entity_dict[i].rfind('_')] == self.entity_dict[j][:self.entity_dict[j].rfind('_')]

    def inference(self, text: str, intent_topk=5):
        if self.model is None:
            raise ValueError(
                "model is not loaded, first call load_model(checkpoint_path)"
            )

        # encode text to token_indices
        tokens = self.model.dataset.encode(text)
        intent_result, entity_result = self.model.forward(tokens.unsqueeze(0))

        # mapping intent result
        rank_values, rank_indicies = torch.topk(
            nn.Softmax(dim=1)(intent_result)[0], k=intent_topk
        )
        intent = {}
        intent_ranking = []
        for i, (value, index) in enumerate(
            list(zip(rank_values.tolist(), rank_indicies.tolist()))
        ):
            intent_ranking.append(
                {"confidence": value, "name": self.intent_dict[index]}
            )

            if i == 0:
                intent["name"] = self.intent_dict[index]
                intent["confidence"] = value

        # mapping entity result
        entities = []

        # except first & last sequnce token whcih indicate BOS or [CLS] token & EOS or [SEP] token
        _, entity_indices = torch.max((entity_result)[0][1:-1, :], dim=1)
        start_idx = -1

        if isinstance(
            self.model.dataset.tokenizer, CharacterEncoder
        ):  # in case of CharacterTokenizer
            entity_indices = entity_indices.tolist()[: len(text)]
            start_idx = -1
            for i, char_idx in enumerate(entity_indices):
                if char_idx != 0 and start_idx == -1:
                    start_idx = i
                elif i > 0 and not self.is_same_entity(i-1, i):
                    end_idx = i
                    entities.append(
                        {
                            "start": max(start_idx, 0),
                            "end": end_idx,
                            "value": text[max(start_idx, 0) : end_idx],
                            "entity": self.entity_dict[entity_indices[i-1]][:self.entity_dict[entity_indices[i-1]].rfind('_')]
                        }
                    )
                    if char_idx == 0:
                        start_idx = -1
                    else:
                        start_idx = i

        else:
            entity_indices = entity_indices.tolist()[:len(text)]
            start_token_position = -1
            for i, entity_idx_value in enumerate(entity_indices):
                if entity_idx_value != 0 and start_token_position == -1:
                    start_token_position = i
                elif i > 0 and not self.is_same_entity(i-1,i):
                    end_token_position = i

                    # except first sequnce token whcih indicate BOS or [CLS] token

                    if type(tokens) == torch.Tensor:
                        tokens = tokens.long().tolist()

                    # find start text position
                    token_idx = tokens[start_token_position + 1]
                    if isinstance(
                        self.model.dataset.tokenizer, WhitespaceEncoder
                    ):  # WhitespaceEncoder
                        token_value = self.model.dataset.tokenizer.index_to_token[
                            token_idx
                        ]
                    elif "KoBertTokenizer" in str(
                        type(self.model.dataset.tokenizer)
                    ):  # KoBertTokenizer
                        token_value = self.model.dataset.tokenizer.idx2token[
                            token_idx
                        ].replace("▁", " ")
                    elif "ElectraTokenizer" in str(
                        type(self.model.dataset.tokenizer)
                    ):  # ElectraTokenizer
                        token_value = self.model.dataset.tokenizer.convert_ids_to_tokens(
                            [token_idx]
                        )[
                            0
                        ].replace(
                            "#", ""
                        )

                    start_position = text.find(token_value.strip())

                    # find end text position
                    token_idx = tokens[end_token_position + 1]
                    if isinstance(
                        self.model.dataset.tokenizer, WhitespaceEncoder
                    ):  # WhitespaceEncoder
                        token_value = self.model.dataset.tokenizer.index_to_token[
                            token_idx
                        ]
                    elif "KoBertTokenizer" in str(
                        type(self.model.dataset.tokenizer)
                    ):  # KoBertTokenizer
                        token_value = self.model.dataset.tokenizer.idx2token[
                            token_idx
                        ].replace("▁", " ")
                    elif "ElectraTokenizer" in str(
                        type(self.model.dataset.tokenizer)
                    ):  # ElectraTokenizer
                        token_value = self.model.dataset.tokenizer.convert_ids_to_tokens(
                            [token_idx]
                        )[
                            0
                        ].replace(
                            "#", ""
                        )

                    end_position = text.find(token_value.strip(), start_position) + len(token_value.strip())

                    entities.append(
                        {
                            "start": start_position,
                            "end": end_position,
                            "value": text[start_position:end_position],
                            "entity": self.entity_dict[entity_indices[i-1]][:self.entity_dict[entity_indices[i-1]].rfind('_')]
                        }
                    )

                    if entity_idx_value == 0:
                        start_token_position = -1
                    else:
                        start_token_position = i

        result = {
            "text": text,
            "intent": intent,
            "intent_ranking": intent_ranking,
            "entities": entities,
        }

        # print (result)

        return result

        # rasa NLU entire result format
        """
        {
            "text": "Hello!",
            "intent": {
                "confidence": 0.6323,
                "name": "greet"
            },
            "intent_ranking": [
                {
                    "confidence": 0.6323,
                    "name": "greet"
                }
            ],
            "entities": [
                {
                    "start": 0,
                    "end": 0,
                    "value": "string",
                    "entity": "string"
                }
            ]
        }
        """
