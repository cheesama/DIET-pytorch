
import torch
from torchnlp.encoders.text import CharacterEncoder, WhitespaceEncoder

class NERDecoder(object):

    def __init__(self, entity_dict:dict, tokenizer):
        self.entity_dict = entity_dict
        self.tokenizer = tokenizer

    def process(self, tokens, entity_indices, text):
        # mapping entity result
        entities = []
        start_idx = -1

       if isinstance(
            self.model.dataset.tokenizer, CharacterEncoder):  # in case of CharacterTokenizer
            entity_indices = entity_indices.tolist()[: len(text)]
            start_idx = -1
            for i, char_idx in enumerate(entity_indices):
                if char_idx != 0 and start_idx == -1:
                    start_idx = i
                elif start_idx >= 0 and not self.is_same_entity(entity_indices[i-1], entity_indices[i]):
                    end_idx = i

                    if self.entity_dict[entity_indices[i-1]] != "O": # ignore 'O' tag
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

            # except first sequnce token whcih indicate BOS or [CLS] token
            if type(tokens) == torch.Tensor:
                tokens = tokens.long().tolist()

            for i, entity_idx_value in enumerate(entity_indices):
                if entity_idx_value != 0 and start_token_position == -1:
                    start_token_position = i
                elif start_token_position >= 0 and not self.is_same_entity(entity_indices[i-1],entity_indices[i]):
                    end_token_position = i

                    # find start text position
                    token_idx = tokens[start_token_position + 1]
                    if isinstance(
                        self.model.dataset.tokenizer, WhitespaceEncoder
                    ):  # WhitespaceEncoder
                        token_value = self.model.dataset.tokenizer.index_to_token[token_idx]
                    elif "KoBertTokenizer" in str(
                        type(self.model.dataset.tokenizer)
                    ):  # KoBertTokenizer
                        token_value = self.model.dataset.tokenizer.idx2token[token_idx].replace("▁", " ")
                    elif "ElectraTokenizer" in str(
                        type(self.model.dataset.tokenizer)
                    ):  # ElectraTokenizer
                        token_value = self.model.dataset.tokenizer.convert_ids_to_tokens([token_idx])[0].replace("#", "")

                    start_position = text.find(token_value.strip())

                    # find end text position
                    token_idx = tokens[end_token_position + 1]
                    if isinstance(self.model.dataset.tokenizer, WhitespaceEncoder):  # WhitespaceEncoder
                        token_value = self.model.dataset.tokenizer.index_to_token[token_idx]
                    elif "KoBertTokenizer" in str(type(self.model.dataset.tokenizer)):  # KoBertTokenizer
                        token_value = self.model.dataset.tokenizer.idx2token[token_idx].replace("▁", " ")
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

                    if self.entity_dict[entity_indices[i-1]] != "O": # ignore 'O' tag
                        entities.append(
                            {
                                "start": start_position,
                                "end": end_position,
                                "value": text[start_position:end_position],
                                "entity": self.entity_dict[entity_indices[i-1]][:self.entity_dict[entity_indices[i-1]].rfind('_')]
                            }
                        )
                        
                        start_token_position = -1

                    if entity_idx_value == 0:
                        start_token_position = -1

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
