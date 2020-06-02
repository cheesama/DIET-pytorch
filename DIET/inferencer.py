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
            self.intent_dict[v] = k

        self.entity_dict = {}
        for k, v in self.model.dataset.entity_dict.items():
            self.entity_dict[v] = k

        logging.info("intent dictionary")
        logging.info(self.intent_dict)
        print()

        logging.info("entity dictionary")
        logging.info(self.entity_dict)

    def inference(self, text: str, intent_topk=5):
        if self.model is None:
            raise ValueError(
                "model is not loaded, first call load_model(checkpoint_path)"
            )

        tokens = self.model.dataset.tokenize(text)
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

        if isinstance(self.model.dataset.tokenizer, CharacterEncoder):  # in case of CharacterTokenizer
            entity_indices = entity_indices.tolist()[: len(text)]
            start_idx = -1
            for i, char_idx in enumerate(entity_indices):
                if char_idx != 0 and start_idx == -1:
                    start_idx = i
                elif i > 0 and entity_indices[i - 1] != entity_indices[i]:
                    end_idx = i
                    entities.append(
                        {
                            "start": max(start_idx, 0),
                            "end": end_idx,
                            "value": text[max(start_idx, 0) : end_idx],
                            "entity": self.entity_dict[entity_indices[i - 1]],
                        }
                    )
                    start_idx = -1

        else:
            entity_indices = entity_indices.tolist()[: len(text)]
            for i, entity_idx in enumerate(entity_indices):
                if entity_idx != 0:
                    token_idx = tokens[i+1] #except first BOS(CLS) token
                    if isinstance(self.model.dataset.tokenizer, WhitespaceEncoder):  # in case of WhitespaceEncoder
                        token_value = self.model.dataset.tokenizer.index_to_token[token_idx]
                    elif "KoBertTokenizer" in str(type(self.model.dataset.tokenizer)):
                        token_value = self.model.dataset.tokenizer.idx2token[token_idx]
                    elif "ElectraTokenizer" in str(type(self.model.dataset.tokenizer)):
                        token_value = self.model.dataset.tokenizer.convert_ids_to_tokens([token_idx])[0]

                    start_position = text.find(token_value)

                    if start_position > 0:
                        entities.append(
                            {
                                "start": start_position,
                                "end": start_position + len(token_value),
                                "value": token_value,
                                "entity": self.entity_dict[entity_indices[i - 1]],
                            }
                        )

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
