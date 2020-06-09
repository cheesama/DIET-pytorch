#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import os
import json
import pandas as pd
from sklearn import metrics
from pycm import ConfusionMatrix
from typing import List, Dict, Sequence
from sklearn import metrics

# TODO: Masked output에 대한 메트릭 함수 구현
def accuracy(pred, label):
    """calculate accuracy using PYCM

    Args:
        pred (numpy.array) : pred label for each batch (batch_size X number of pred classes)
        label (numpy.array) : label for each batch (batch_size X number of classes)
    
    Returns:
        accuracy (float) : overall accuracy value

    """
    assert type(pred) == type(label), "each type should be same"

    pred = pred.reshape(-1)
    label = label.reshape(-1)

    cm = ConfusionMatrix(pred, label)
    return cm.Overall_ACC


def masked_accuracy(pred, label, mask):
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    mask = mask.reshape(-1)

    n_total = sum(mask)
    n_correct = sum((pred == label) & mask)
    accu = n_correct / n_total

    return accu

def confusion_matrix(pred: list, label:list, label_index:dict, file_name:str=None, output_dir='results'):
    cm = ConfusionMatrix(pred, label)
    cm.relabel(mapping=label_index)
    cm_matrix = cm.matrix
    cm_normalized_matrix = cm.normalized_matrix

    if file_name is None:
        file_name = 'confusion_matrix.json'
    
    normalized_file_name = file_name.replace('.', '_normalized.')

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, file_name), 'w') as fp:
            json.dump(cm_matrix, fp, indent=4)
        
        # with open(os.path.join(output_dir, normalized_file_name), 'w') as fp:
        #     json.dump(cm_normalized_matrix, fp, indent=4)
    
    return cm_matrix


def pred_report(inequal_dict: dict, cm_matrix: dict, file_name: None, output_dir='results'):
    if file_name is None:
        file_name = './result/prediction_reports.md'

    output = """"""
    intent_list = list(cm_matrix.keys())
    for intent in intent_list:
        tmp_data = cm_matrix[intent]
        tmp_df = pd.DataFrame(list(tmp_data.items()),
                              columns=['intent', 'count'])
        total_count = tmp_df['count'].sum()
        output += "## Prediction intent: {} / total_count: {}".format(
            intent, total_count)
        output += "\n"
        tmp_df = tmp_df.sort_values(
            by='count', ascending=False).reset_index(drop=True)
        tmp_df = tmp_df[tmp_df['count'] > 0]
        tmp_df.to_dict()

        if intent in inequal_dict.keys():
            tmp_inequal_dict = inequal_dict[intent]
        
            for i in range(tmp_df.shape[0]):
                tmp = tmp_df.iloc[i].tolist()
                actual_intent = tmp[0]
                actual_cnt = tmp[1]
                tmp_text = []
                for x in tmp_inequal_dict:
                    if x['target'] == actual_intent:
                        tmp_text.append(x)

                actual_ratio = round(actual_cnt / total_count, 2)
                output += "- Actual Intent: {} count:{} ratio:{}".format(
                    actual_intent, actual_cnt, actual_ratio)
                output += "\n"
                if len(tmp_text) > 0:
                    for t in tmp_text:
                        output += "   - Text: {}  prob: {}".format(
                            t['text'], t['prob'])
                        output += "\n"
        elif tmp_df.shape[0] > 0:
            output += "- All predictions match actual intent."
            output += "\n"

        else:
            output += "- None of predicted values exist."
            output += "\n"

        output += "\n"
        output += "\n"

    text_file = open(os.path.join(output_dir, file_name), "w")
    _ = text_file.write(output)
    text_file.close()

def show_classification_metrics(
    pred, label, use_matric=None, label_index=None, matrics_list=None, display=True
):
    """calc metrics for classification model using PYCM

    Args:
        pred (numpy.array) : pred label for each batch (batch_size X number of pred classes)
        label (numpy.array) : label for each batch (batch_size X number of classes)
        label_index (dict) : class name (default=None)
        matrics_list (list) : add matrics name (refer to the pycm metrics list) (defailt=None)
        display (bool) :  Whether to print the overall result (default=True)
    
    Returns:
        metrics (dict) : contain the 2-level result (overall_stat, class_stat)

    """
    # pred = pred.reshape(-1)
    # label = label.reshape(-1)

    cm = ConfusionMatrix(pred, label)
    if label_index is not None:
        cm.relabel(mapping=label_index)

    default_matrics_list = cm.recommended_list
    if matrics_list is not None:
        default_matrics_list.extend(matrics_list)

    if display:
        cm.stat(summary=True)
        print("[Matrix]")
        cm.print_matrix()
        print("[Normalized Matrix]")
        cm.print_normalized_matrix()

    overall_stat = cm.overall_stat
    class_stat = cm.class_stat

    filter_overall_stat = {
        k: v for k, v in overall_stat.items() if k in default_matrics_list
    }
    filter_class_stat = {
        k: v for k, v in class_stat.items() if k in default_matrics_list
    }
    output = dict()
    output["overall_stat"] = filter_overall_stat
    output["class_stat"] = filter_class_stat
    return output


def show_rasa_metrics(pred, label, labels=None, target_names=None, output_dir='results', file_name=None):

    output = metrics.classification_report(label, pred, labels=labels,
                                           target_names=target_names, output_dict=True)
                                           
    if file_name is None:
        file_name = 'reports.json'
        
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, file_name), 'w') as fp:
            json.dump(output, fp, indent=4)

    return output


class Entity_Matrics:
    def __init__(self, sents_true_labels: Sequence[Sequence[Dict]], sents_pred_labels:Sequence[Sequence[Dict]]):
        self.sents_true_labels = sents_true_labels
        self.sents_pred_labels = sents_pred_labels 
        self.types = set(entity['entity'] for sent in sents_true_labels for entity in sent)
        self.confusion_matrices = {type: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for type in self.types}
        self.scores = {type: {'p': 0, 'r': 0, 'f1': 0} for type in self.types}

    def cal_confusion_matrices(self) -> Dict[str, Dict]:
        """Calculate confusion matrices for all sentences."""
        for true_labels, pred_labels in zip(self.sents_true_labels, self.sents_pred_labels):
            for true_label in true_labels: 
                entity_type = true_label['entity']
                prediction_hit_count = 0 
                for pred_label in pred_labels:
                    if pred_label['entity'] != entity_type:
                        continue
                    if pred_label['start'] == true_label['start'] and pred_label['end'] == true_label['end'] and pred_label['value'] == true_label['value']: # TP
                        self.confusion_matrices[entity_type]['TP'] += 1
                        prediction_hit_count += 1
                    elif ((pred_label['start'] == true_label['start']) or (pred_label['end'] == true_label['end'])) and pred_label['value'] != true_label['value']: # boundry error, count FN, FP
                        self.confusion_matrices[entity_type]['FP'] += 1
                        self.confusion_matrices[entity_type]['FN'] += 1
                        prediction_hit_count += 1
                if prediction_hit_count != 1: # FN, model cannot make a prediction for true_label
                    self.confusion_matrices[entity_type]['FN'] += 1
                prediction_hit_count = 0 # reset to default

    def cal_scores(self) -> Dict[str, Dict]:
        """Calculate precision, recall, f1."""
        confusion_matrices = self.confusion_matrices 
        scores = {type: {'p': 0, 'r': 0, 'f1': 0} for type in self.types}
        
        for entity_type, confusion_matrix in confusion_matrices.items():
            if confusion_matrix['TP'] == 0 and confusion_matrix['FP'] == 0:
                scores[entity_type]['p'] = 0
            else:
                scores[entity_type]['p'] = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])

            if confusion_matrix['TP'] == 0 and confusion_matrix['FN'] == 0:
                scores[entity_type]['r'] = 0
            else:
                scores[entity_type]['r'] = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN']) 

            if scores[entity_type]['p'] == 0 or scores[entity_type]['r'] == 0:
                scores[entity_type]['f1'] = 0
            else:
                scores[entity_type]['f1'] = 2*scores[entity_type]['p']*scores[entity_type]['r'] / (scores[entity_type]['p']+scores[entity_type]['r'])  
        self.scores = scores

    def print_confusion_matrices(self):
        for entity_type, matrix in self.confusion_matrices.items():
            print(f"{entity_type}: {matrix}")

    def print_scores(self):
        for entity_type, score in self.scores.items():
            print(f"{entity_type}: f1 {score['f1']:.4f}, precision {score['p']:.4f}, recall {score['r']:.4f}")
        
    def cal_micro_avg(self):
        sum_TP = 0
        sum_FP = 0
        sum_FN = 0
        support = 0
        for k, v in self.confusion_matrices.items():
            sum_TP += v['TP']
            sum_FP += v['FP']
            sum_FN += v['FN']
            support += np.array(list(self.confusion_matrices[k].values())).sum().item()
        precision = sum_TP / (sum_TP + sum_FP)
        recall = sum_TP / (sum_TP + sum_FN)
        f1 = 2*(precision * recall / (precision + recall))
        self.micro_avg = dict()
        self.micro_avg['precision'] = precision
        self.micro_avg['recall'] = recall
        self.micro_avg['f1-score'] = f1
        self.micro_avg['support'] = support
    
    def cal_macro_avg(self):
        precision = []
        recall = []
        support = 0
        for k, v in self.scores.items():
            precision.append(v['p'])
            recall.append(v['r'])
        for k, v in self.confusion_matrices.items():
            support += np.array(list(self.confusion_matrices[k].values())).sum().item()
        precision = np.array(precision).mean()
        recall = np.array(recall).mean()
        f1 = 2*(precision * recall / (precision + recall))
        self.macro_avg = dict()
        self.macro_avg['precision'] = precision
        self.macro_avg['recall'] = recall
        self.macro_avg['f1-score'] = f1
        self.macro_avg['support'] = support
    
    def cal_weight_avg(self):
        tp = []
        fp = []
        fn = []
        weight = []
        support = 0
        for k, v in self.confusion_matrices.items():
            tp.append(v['TP'])
            fp.append(v['FP'])
            fn.append(v['FN'])
            weight.append(np.array(list(v.values())).sum().item())
            support += np.array(list(self.confusion_matrices[k].values())).sum().item()

        weight = np.array(weight) / np.array(weight).sum()
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)
        precision = (weight * tp).sum() / ((weight * tp).sum() + (weight * fp).sum())
        recall = (weight * tp).sum() / ((weight * tp).sum() + (weight * fn).sum())
        f1 = 2*(precision * recall / (precision + recall))
        self.weight_avg = dict()
        self.weight_avg['precision'] = precision.item()
        self.weight_avg['recall'] = recall.item()
        self.weight_avg['f1-score'] = f1.item()
        self.weight_avg['support'] = support
    
    def generate_report(self):
        self.cal_confusion_matrices()
        self.cal_scores()
        self.cal_micro_avg()
        self.cal_macro_avg()
        self.cal_weight_avg()
        
        report = dict()
        for k, v in self.scores.items():
            report[k] = dict()
            report[k]['precision'] = v['p']
            report[k]['recall'] = v['r']
            report[k]['f1-score'] = v['f1']
            report[k]['support'] = np.array(list(self.confusion_matrices[k].values())).sum().item()
        report['micro avg'] = self.micro_avg
        report['macro avg'] = self.macro_avg
        report['weighted avg'] = self.weight_avg
        return report