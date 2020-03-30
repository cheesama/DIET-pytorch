from collections import OrderedDict
from tqdm import tqdm

import torch
import re


class RasaIntentEntityDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.intent_dict = {}
        self.entity_dict = {}  # based on XO tagging(one entity has one class)
        self.dataset = []

        current_intent_focus = ""

        with open(file_path, "r", encoding="utf-8") as nluFile:
            for line in tqdm(nluFile.readlines(), desc='Extracting Intent & Entity in NLU markdown files...'):
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
                        each_data_dict["text"] = text
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pass
