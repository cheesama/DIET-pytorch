from .DIET_lightning_model import DualIntentEntityTransformer

import torch

model = None


def load_model(checkpoint_path: str):
    model = DualIntentEntityTransformer.load_from_checkpoint(
        checkpoint_path, data_file_path=None
    )

    return model


def inference(text: str):
    if model is None:
        raise ValueError("model is not loaded, first call load_model(checkpoint_path)")

    tokens = model.dataset.tokenize(text)
    intent_result, entity_result = model.forward(tokens)

    # model.dataset.intent_dict
    # model.dataset.entity_dict

    # rasa NLU result format
    """
    {
        "entities": [
            {
                "start": 0,
                "end": 0,
                "value": "string",
                "entity": "string",
                "confidence": 0
            }
        ],
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
        "text": "Hello!"
    }
    """
