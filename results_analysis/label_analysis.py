# -*- coding: utf-8 -*-
# -----------------------
# Author : Wen Ting, Choi
# Date : 2023-08-09 
# -----------------------

"""Methods used for results analysis by label
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def accuracy_score_by_label(df, label):
    _df = df[df['label'] == label].copy()
    return sum(_df['correctly_classified'])/len(_df)

def get_accuracy_scores(df, labels):
    return np.array([accuracy_score_by_label(df, label) for label in labels])

def get_statistics_for_group(df, labels=[0,1,2]):
    y = df['label']
    y_pred = df['pred']

    accuracy = get_accuracy_scores(df, labels)
    precision = precision_score(y, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y, y_pred, labels=labels, average=None)
    f1 = f1_score(y, y_pred, labels=labels, average=None)

    return labels, accuracy, precision, recall, f1

def gen_statistics(df):
    grouped = df.groupby(['train_dataset', 'test_dataset', 'model'])
    for (train_dataset, test_dataset, model), _df in grouped:
        labels, accuracy, precision, recall, f1 = get_statistics_for_group(_df) #each metrics are returned in a list of len=3 
        for i in range(len(labels)):
            yield (
                model,
                train_dataset,
                test_dataset,
                labels[i], 
                accuracy[i], 
                precision[i], 
                recall[i], 
                f1[i]
            )

def compute_statistics(df):
    """Compute various statistics by by label
    """
    stats = list(gen_statistics(df))
    stats = pd.DataFrame(stats, columns=
        ['model', 'train_dataset', 'test_dataset', 
         'labels', 'accuracy', 'precision', 'recall', 'f1']
    )
    return stats

def prettify_table(stats, metrics='accuracy'):
    """Prettify dataframe for analysis and presentation
    """
    # convert metrics to percentage
    cols = ['accuracy', 'precision', 'recall', 'f1']
    for col in cols:
        stats[col] = stats[col]*100

    # replace labels
    stats['labels'].replace({
        0: 'SUPPORTS',
        1: 'REFUTES',
        2: 'NOT ENOUGH INFO'
    }, inplace=True)

    stats['train_dataset'].replace({
        'FEVER': 'Fine-tuned on FEVER',
        'PUBHEALTH': 'Fine-tuned on PUBHEALTH',
        'CLIMATE-FEVER': 'Fine-tuned on CLIMATE-FEVER'
    }, inplace=True)

    stats['test_dataset'].replace({
        'fever': 'FEVER',
        'pubhealth': 'PUBHEALTH',
        'climate': 'CLIMATE-FEVER'
    }, inplace=True)
    
    # pivot by model for better presentation
    stats_pivot = stats.pivot(index=['train_dataset', 'test_dataset', 'labels'], columns='model', values=metrics)
    stats_pivot = stats_pivot[['BERT', 'RoBERTa', 'ALBERT', 'SciBERT', 'BioBERT']].\
        reindex(['Fine-tuned on FEVER', 'Fine-tuned on PUBHEALTH', 'Fine-tuned on CLIMATE-FEVER'], level=0).\
        reindex(['FEVER', 'PUBHEALTH', 'CLIMATE-FEVER'], level=1).\
        reindex(['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'], level=2)
    
    return stats_pivot