from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from .metrics import show_rasa_metrics, confusion_matrix, pred_report
from DIET.decoder import NERDecoder  
import torch
import logging
import numpy as np

def show_intent_report(dataset, pl_module, tokenizer, file_name=None, output_dir=None, cuda=True):
    ##generate rasa performance matrics
    # text = []
    preds = np.array([])
    targets = np.array([])
    logits = np.array([])
    label_dict = dict()
    pl_module.model.eval()
    for k, v in pl_module.intent_dict.items():
        label_dict[int(k)] = v
    dataloader = DataLoader(dataset, batch_size=32)

    for batch in tqdm(dataloader, desc="load intent dataset"):
        #dataset follows RasaIntentEntityDataset which defined in this package
        input_ids, intent_idx, entity_idx, text = batch
        model =  pl_module.model
        if cuda > 0:
            input_ids = input_ids.cuda()
            model = model.cuda()
        intent_pred, entity_pred = model.forward(input_ids)
        y_label = intent_pred.argmax(1).cpu().numpy()
        preds = np.append(preds, y_label)
        targets = np.append(targets, intent_idx.cpu().numpy())
        
        logit = intent_pred.detach().cpu()
        softmax = torch.nn.Softmax(dim=-1)
        logit = softmax(logit).numpy()
        logits = np.append(logits, logit.max(-1))
    
    preds = preds.astype(int)
    targets = targets.astype(int)

    labels = list(label_dict.keys())
    target_names = list(label_dict.values())
    
    report = show_rasa_metrics(pred=preds, label=targets, labels=labels, target_names=target_names, file_name=file_name, output_dir=output_dir)

def show_entity_report(dataset, pl_module, file_name=None, output_dir=None, cuda=True):
    
    ##generate rasa performance matrics
    tokenizer = dataset.tokenizer
    text = []
    label_dict = dict()
    pl_module.model.eval()
    for k, v in pl_module.entity_dict.items():
        label_dict[int(k)] = v

    decoder = NERDecoder(label_dict, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32)

    preds = list()
    targets = list()
    labels = set()

    for batch in tqdm(dataloader, desc="load entity dataset"):
        input_ids, intent_idx, entity_idx, token = batch
        text.extend(token)
        if cuda > 0:
            input_ids = input_ids.cuda()
        _, entity_result = pl_module.model.forward(input_ids)

        entity_result = entity_result.detach().cpu()
        _, entity_indices = torch.max(entity_result, dim=-1)



        for i in range(entity_idx.shape[0]):
            decode_original = decoder.process(input_ids[i].cpu().numpy(), entity_idx[i].numpy())
            decode_pred = decoder.process(input_ids[i].cpu().numpy(), entity_indices[i].numpy())
            targets.append(decode_original)
            preds.append(decode_pred)

            # for origin in decode_original:
            #     labels.add(origin['entity'])
            #     find_idx = 0
            #     for pred in decode_pred:
            #         if origin['start'] == pred['start'] and origin['end'] == pred['end']:
            #             preds.append(origin['entity'])
            #             targets.append(origin['entity'])
            #             find_idx += 1
            #     if find_idx == 0:
            #          preds.append('No_Entity')
            #          targets.append(origin['entity'])


    report = show_entity_metrics(pred=preds, label=targets, file_name=file_name, output_dir=output_dir)


