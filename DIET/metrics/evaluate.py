import logging
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
# from electra_diet.metrics import show_rasa_metrics
from .metrics import show_rasa_metrics, confusion_matrix, pred_report


logger = logging.getLogger(__name__)


def show_intent_report(dataset, pl_module, file_name=None, output_dir=None, cuda=True):
    ##generate rasa performance matrics
    tokenizer = pl_module.dataset.tokenizer
    text = []
    preds = np.array([])
    targets = np.array([])
    logits = np.array([])
    label_dict = dict()
    for k, v in pl_module.intent_dict.items():
        label_dict[int(k)] = v
    
    dataloader = DataLoader(dataset, batch_size=32)
    for batch in dataloader:
        inputs, intent_idx, entity_idx = batch
        (input_ids, token_type_ids) = inputs
        token = get_token_to_text(tokenizer, input_ids)
        text.extend(token)
        if cuda > 0:
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
        intent_pred, entity_pred = pl_module.model(input_ids, token_type_ids)
        y_label = intent_pred.argmax(1).cpu().numpy()
        preds = np.append(preds, y_label)
        targets = np.append(targets, intent_idx.cpu().numpy())
        
        logit = intent_pred.detach().cpu()
        softmax = torch.nn.Softmax(dim=-1)
        logit = softmax(logit).numpy()
        logits = np.append(logits, logit)
    
    preds = preds.astype(int)
    targets = targets.astype(int)

    labels = list(label_dict.keys())
    target_names = list(label_dict.values())
    
    report = show_rasa_metrics(pred=preds, label=targets, labels=labels, target_names=target_names, file_name=file_name)
    
    ##generate confusion matrix
    inequal_index = np.where(preds != targets)[0]
    inequal_dict = dict()
    for i in trange(inequal_index.shape[0]):
        idx = inequal_index[i].item()
        pred = preds[idx]
        if label_dict[pred] not in inequal_dict.keys():
            inequal_dict[label_dict[pred]] = []
        tmp_dict = dict()
        tmp_dict['target'] = label_dict[targets[idx]]
        tmp_dict['prob'] = round(logits[idx], 3)
        tmp_dict['text'] = text[idx]
        inequal_dict[label_dict[pred]].append(tmp_dict)
    
    cm_matrix = confusion_matrix(
            pred=preds, label=targets, label_index=label_dict, file_name=file_name, output_dir=output_dir)
    
    pred_report(inequal_dict, cm_matrix, file_name=file_name.replace(
            '.json', '.md'),  output_dir=output_dir)
    
def get_token_to_text(tokenizer, data):
    values = []
    for token in data:
        values.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([x for x in token if x >4])))
    return values
