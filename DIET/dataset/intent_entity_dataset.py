from collections import OrderedDict
from tqdm import tqdm
from typing import List

# related to pretrained tokenizer & model
from transformers import ElectraModel, ElectraTokenizer
from kobert_transformers import get_kobert_model, get_distilkobert_model
from kobert_transformers import get_tokenizer as kobert_tokenizer

from torchnlp.encoders.text import CharacterEncoder, WhitespaceEncoder

import torch
import numpy as np
import re

class RasaIntentEntityDataset(torch.utils.data.Dataset):
    """
    RASA NLU markdown file lines based Custom Dataset Class

    Dataset Example in nlu.md

    ## intent:intent_데이터_자동_선물하기_멀티턴                <- intent name
    - T끼리 데이터 주기적으로 보내기                            <- utterance without entity
    - 인터넷 데이터 [달마다](Every_Month)마다 보내줄 수 있어?    <- utterance with entity
    
    """

    def __init__(
        self,
        markdown_lines: List[str],
        tokenizer,
        seq_len=128,
    ):
        self.intent_dict = {}
        self.entity_dict = {}
        self.entity_dict[
            "O"
        ] = 0  # based on XO tagging(one entity_type has assigned to one class)

        self.dataset = []
        self.seq_len = seq_len

        intent_value_list = []
        entity_type_list = []

        current_intent_focus = ""

        text_list = []

        for line in tqdm(
            markdown_lines,
            desc="Organizing Intent & Entity dictionary in NLU markdown file ...",
        ):
            if len(line.strip()) < 2:
                current_intent_focus = ""
                continue

            if "## " in line:
                if "intent:" in line:
                    intent_value_list.append(line.split(":")[1].strip())
                    current_intent_focus = line.split(":")[1].strip()
                else:
                    current_intent_focus = ""

            else:
                if current_intent_focus != "":
                    text = line[2:].strip()

                    for type_str in re.finditer(r"\([a-zA-Z_1-2]+\)", text):
                        entity_type = (
                            text[type_str.start() + 1 : type_str.end() - 1]
                            .replace("(", "")
                            .replace(")", "")
                        )
                        entity_type_list.append(entity_type)

        intent_value_list = sorted(intent_value_list)
        for intent_value in intent_value_list:
            if intent_value not in self.intent_dict.keys():
                self.intent_dict[intent_value] = len(self.intent_dict)

        entity_type_list = sorted(entity_type_list)
        for entity_type in entity_type_list:
            if entity_type not in self.entity_dict.keys():
                self.entity_dict[entity_type] = len(self.entity_dict)

        current_intent_focus = ""

        for line in tqdm(
            markdown_lines, desc="Extracting Intent & Entity in NLU markdown files...",
        ):
            if len(line.strip()) < 2:
                current_intent_focus = ""
                continue

            if "## " in line:
                if "intent:" in line:
                    current_intent_focus = line.split(":")[1].strip()
                else:
                    current_intent_focus = ""
            else:
                if current_intent_focus != "":  # intent & entity sentence occur case
                    text = line[2:]

                    entity_value_list = []
                    for value in re.finditer(r"\[(.*?)\]", text):
                        entity_value_list.append(
                            text[value.start() + 1 : value.end() - 1]
                            .replace("[", "")
                            .replace("]", "")
                        )

                    entity_type_list = []
                    for type_str in re.finditer(r"\([a-zA-Z_1-2]+\)", text):
                        entity_type = (
                            text[type_str.start() + 1 : type_str.end() - 1]
                            .replace("(", "")
                            .replace(")", "")
                        )
                        entity_type_list.append(entity_type)

                    text = re.sub(r"\([a-zA-Z_1-2]+\)", "", text)  # remove (...) str
                    text = text.replace("[", "").replace(
                        "]", ""
                    )  # remove '[',']' special char

                    text_list.append(text)

                    each_data_dict = {}
                    each_data_dict["text"] = text.strip()
                    each_data_dict["intent"] = current_intent_focus
                    each_data_dict["intent_idx"] = self.intent_dict[
                        current_intent_focus
                    ]
                    each_data_dict["entities"] = []

                    for value, type_str in zip(entity_value_list, entity_type_list):
                        try:
                            for entity in re.finditer(value, text):
                                each_data_dict["entities"].append(
                                    {
                                        "start": entity.start(),
                                        "end": entity.end(),
                                        "entity": type_str,
                                        "value": value,
                                        "entity_idx": self.entity_dict[type_str],
                                    }
                                )

                        except Exception as ex:
                            print(f"error occured : {ex}")
                            print(f"value: {value}")
                            print(f"text: {text}")

                    self.dataset.append(each_data_dict)

        print(f"Intents: {self.intent_dict}")
        print(f"Entities: {self.entity_dict}")

        if "KoBertTokenizer" in str(type(tokenizer)):
            self.tokenizer = tokenizer
            self.pad_token_id = 1
            self.unk_token_id = 0
            self.eos_token_id = 3 #[SEP] token
            self.bos_token_id = 2 #[CLS] token

        elif "ElectraTokenizer" in str(type(tokenizer)):
            self.tokenizer = tokenizer
            self.pad_token_id = 0
            self.unk_token_id = 1
            self.eos_token_id = 3 #[SEP] token
            self.bos_token_id = 2 #[CLS] token

        else:
            if tokenizer == 'char':
                self.tokenizer = CharacterEncoder(text_list)
            elif tokenizer == 'space':
                self.tokenizer = WhitespaceEncoder(text_list)

            # torchnlp base special token indices
            self.pad_token_id = 0
            self.unk_token_id = 1
            self.eos_token_id = 2
            self.bos_token_id = 3

    def tokenize(self, text: str, padding: bool = True, return_tensor: bool = True):
        tokens = self.tokenizer.encode(text)
        if type(tokens) == list:
            tokens = torch.tensor(tokens)

        # kobert_tokenizer & koelectra tokenize append [CLS](2) token to start and [SEP](3) token to end
        if isinstance(self.tokenizer, CharacterEncoder) or isinstance(
            self.tokenizer, WhitespaceEncoder
        ):
            bos_tensor = torch.tensor([self.bos_token_id])
            eos_tensor = torch.tensor([self.eos_token_id])
            tokens = torch.cat((bos_tensor, tokens, eos_tensor), 0)

        if padding:
            if len(tokens) > self.seq_len:
                tokens = tokens[: self.seq_len]
            else:
                pad_tensor = torch.tensor(
                    [self.pad_token_id] * (self.seq_len - len(tokens))
                )
                tokens = torch.cat((tokens, pad_tensor), 0)

        if return_tensor:
            return tokens
        else:
            return tokens.numpy()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokens = self.tokenize(self.dataset[idx]["text"])

        intent_idx = torch.tensor([self.dataset[idx]["intent_idx"]])

        entity_idx = np.array(self.seq_len * [self.pad_token_id])

        for entity_info in self.dataset[idx]["entities"]:
            if isinstance(self.tokenizer, CharacterEncoder):
                # Consider [CLS](bos) token
                for i in range(entity_info["start"] + 1, entity_info["end"] + 2):
                    entity_idx[i] = entity_info["entity_idx"]

            elif isinstance(self.tokenizer, WhitespaceEncoder):
                ##check whether entity value is include in space splitted token
                for token_seq, token_value in enumerate(tokens):
                    # Consider [CLS](bos) token
                    if token_seq == 0:
                        continue

                    for entity_seq, entity_info in enumerate(
                        self.dataset[idx]["entities"]
                    ):
                        if (
                            entity_info["value"]
                            in self.tokenizer.vocab[token_value.item()]
                        ):
                            entity_idx[token_seq] = entity_info["entity_idx"]
                            break

            elif "KoBertTokenizer" in str(type(self.tokenizer)):
                ##check whether entity value is include in splitted token
                for token_seq, token_value in enumerate(tokens):
                    # Consider [CLS](bos) token
                    if token_seq == 0:
                        continue

                    for entity_seq, entity_info in enumerate(
                        self.dataset[idx]["entities"]
                    ):
                        if (
                            self.tokenizer.idx2token[token_value.item()]
                            in entity_info["value"]
                        ):
                            entity_idx[token_seq] = entity_info["entity_idx"]
                            break

            elif "ElectraTokenizer" in str(type(self.tokenizer)):
                ##check whether entity value is include in splitted token
                for token_seq, token_value in enumerate(tokens):
                    # Consider [CLS](bos) token
                    if token_seq == 0:
                        continue

                    for entity_seq, entity_info in enumerate(
                        self.dataset[idx]["entities"]
                    ):
                        if (
                            self.tokenizer.convert_ids_to_tokens([token_value.item()])[
                                0
                            ]
                            in entity_info["value"]
                        ):
                            entity_idx[token_seq] = entity_info["entity_idx"]
                            break

        entity_idx = torch.from_numpy(entity_idx)

        return tokens, intent_idx, entity_idx

    def get_intent_idx(self):
        return self.intent_dict

    def get_entity_idx(self):
        return self.entity_dict

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_seq_len(self):
        return self.seq_len
