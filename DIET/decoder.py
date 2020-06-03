
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

        if isinstance(self.tokenizer, CharacterEncoder):  # in case of CharacterTokenizer
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
                    if isinstance(self.tokenizer, WhitespaceEncoder):  # in case of WhitespaceEncoder
                        token_value = self.tokenizer.index_to_token[token_idx]
                    elif "KoBertTokenizer" in str(type(self.tokenizer)):
                        token_value = self.tokenizer.idx2token[token_idx]
                    elif "ElectraTokenizer" in str(type(self.tokenizer)):
                        token_value = self.tokenizer.convert_ids_to_tokens([token_idx])[0]

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
        
        return entities
                
# def decode_text(tokenizer, token):
#     replace_index = []
#     if isinstance(tokenizer, CharacterEncoder):
#         replace_index.append('<s>')
#         replace_index.append('</s>')
#         replace_index.append('<pad>')

#     if len(replace_index) > 0:
#         decode = tokenizer.decode(token)
#         for r in replace_index:
#             decode = decode.replace(r, '')  
#     else:
#         decode = tokenizer.decode(token, skip_special_tokens=True)

#     return decode

