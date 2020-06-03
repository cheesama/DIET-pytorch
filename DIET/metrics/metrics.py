#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import os
import json
import pandas as pd
from sklearn import metrics
from pycm import ConfusionMatrix
# ==================================================
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

