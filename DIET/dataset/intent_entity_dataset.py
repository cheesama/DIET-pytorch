from collections import OrderedDict
from tqdm import tqdm
from torchnlp.encoders.text import CharacterEncoder

import torch
import numpy as np
import re


class RasaIntentEntityDataset(torch.utils.data.Dataset):
    """
    RASA NLU markdown file based Custom Dataset Class

    Dataset Example in nlu.md

    ## intent:intent_데이터_자동_선물하기_멀티턴                <- intent name
    - T끼리 데이터 주기적으로 보내기                            <- utterance without entity
    - 인터넷 데이터 [달마다](Every_Month)마다 보내줄 수 있어?    <- utterance with entity
    
    """

    def __init__(
        self,
        file_path,
        seq_len=128,
        pad_token_id=0,
        unk_token_id=1,
        eos_token_id=2,
        bos_token_id=3,
    ):
        self.intent_dict = {}
        self.entity_dict = {}
        self.entity_dict[
            "O"
        ] = 0  # based on XO tagging(one entity_type has assigned to one class)

        self.dataset = []
        self.seq_len = seq_len

        # following torchnlp encoder preset
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

        current_intent_focus = ""

        with open(file_path, "r", encoding="utf-8") as nluFile:
            for line in tqdm(
                nluFile.readlines(),
                desc="Extracting Intent & Entity in NLU markdown files...",
            ):
                if len(line.strip()) < 2:
                    continue

                if "## " in line:
                    if "intent:" in line:
                        current_intent_focus = line.split(":")[1].strip()

                        if current_intent_focus not in self.intent_dict.keys():
                            self.intent_dict[current_intent_focus] = len(
                                self.intent_dict.keys()
                            )

                    else:
                        current_intent_focus = ""
                else:
                    if (
                        current_intent_focus != ""
                    ):  # intent & entity sentence occur case
                        text = line[2:]

                        entity_value_list = []
                        for value in re.finditer(r"\[[^)]*\]", text):
                            entity_value_list.append(
                                text[value.start() + 1 : value.end() - 1]
                            )

                        entity_type_list = []
                        for type_str in re.finditer(r"\([^)]*\)", text):
                            entity_type = text[
                                type_str.start() + 1 : type_str.end() - 1
                            ]
                            entity_type_list.append(entity_type)

                            if entity_type not in self.entity_dict.keys():
                                self.entity_dict[entity_type] = len(
                                    self.entity_dict.keys()
                                )

                        text = re.sub(r"\([^)]*\)", "", text)
                        text = text.replace("[", "").replace("]", "")

                        each_data_dict = {}
                        each_data_dict["text"] = text.strip()
                        each_data_dict["intent"] = current_intent_focus
                        each_data_dict["intent_idx"] = self.intent_dict[
                            current_intent_focus
                        ]
                        each_data_dict["entities"] = []

                        for value, type_str in zip(entity_value_list, entity_type_list):
                            for entity in re.finditer(value, text):
                                each_data_dict["entities"].append(
                                    {
                                        "start": entity.start(),
                                        "end": entity.end(),
                                        "entity": type_str,
                                        "entity_idx": self.entity_dict[type_str],
                                    }
                                )

                        self.dataset.append(each_data_dict)

        # encoder(tokenizer) definition
        self.encoder = CharacterEncoder([data["text"] for data in self.dataset])

    def tokenize(self, text: str, padding: bool = True, return_tensor: bool = True):
        # bos_token=3, eos_token=2, unk_token=1, pad_token=0
        tokens = self.encoder.encode(text)
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

        entity_idx = np.zeros(self.seq_len)
        for entity_info in self.dataset[idx]["entities"]:
            for i in range(entity_info["start"], entity_info["end"] + 1):
                entity_idx[i] = entity_info["entity_idx"]
        entity_idx = torch.from_numpy(entity_idx)

        return tokens, intent_idx, entity_idx

    def get_intent_idx(self):
        return self.intent_dict

    def get_entity_idx(self):
        return self.entity_dict

    def get_vocab_size(self):
        return len(self.encoder.vocab)

    def get_seq_len(self):
        return self.seq_len
