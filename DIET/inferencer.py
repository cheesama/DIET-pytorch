from .DIET_lightning_model import DualIntentEntityTransformer

import torch
import torch.nn as nn

model = None
intent_dict = {}
entity_dict = {}

def load_model(checkpoint_path: str):
    model = DualIntentEntityTransformer.load_from_checkpoint(checkpoint_path)

    model.model.eval()
    model.model.freeze()

    for k, v in model.dataset.intent_dict:
        intent_dict[v] = k

    for k, v in model.dataset.entity_dict:
        entity_dict[v] = k

    return model

def inference(text: str, intent_topk=5):
    if model is None:
        raise ValueError("model is not loaded, first call load_model(checkpoint_path)")

    tokens = model.dataset.tokenize(text)
    intent_result, entity_result = model.forward(tokens)

    # mapping intent result
    rank_values, rank_indicies = torch.topk(nn.Softmax(dim=1)(intent_result)[0], k=intent_topk)
    intent = {}
    intent_ranking = []
    for i, (value, index) in enumerate(list(zip(rank_values.tolist(), rank_indicies.tolist()))):
        intent_ranking.append({'confidence': value, 'name': intent_dict[index]})

        if i == 0:
            intent['name'] = intent_dict[index]
            intent['confidence'] = value

    # mapping entity result
    entities=[]
    # except first sequnce token whcih indicate BOS token
    _, entity_indices = torch.max((entity_result)[0][:,1:,:], dim=1)
    entity_indices.tolist()

    start_idx = 0
    for i, char_idx in enumerate(entity_indices):
        if i > 0 and entity_indices[i-1] != entity_indices[i]:
            end_idx = i-1
            entities.append({'start': start_idx, 'end':end_idx, 'value':text[1 + start_idx:end_idx + 1], 'entity': entity_dict[entity_indices[i-1]]})
            start_idx = i

        if i == len(entity_indices)-1 and entity_indices[i-1] == entity_indices[i]:
            end_idx = i
            entities.append({'start': start_idx, 'end':end_idx, 'value':text[1 + start_idx:end_idx + 1], 'entity': entity_dict[entity_indices[i-1]]})

    return {
        "text": text,
        "intent": intent,
        "intent_ranking": intent_ranking,
        "entities": entity_dict
    }

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
