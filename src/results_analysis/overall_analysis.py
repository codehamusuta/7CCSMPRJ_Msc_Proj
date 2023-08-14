# -*- coding: utf-8 -*-
# -----------------------
# Author : Wen Ting, Choi
# Date : 2023-08-09 
# -----------------------

"""Methods used for analysis at aggregate level
"""

import numpy as np
import pandas as pd

def compute_accuracies(df):
    """Compute overall accuracy
    """
    acc = df.groupby(['train_dataset','test_dataset', 'model'])['correctly_classified'].mean()*100
    acc = acc.reset_index()

    return acc   

def prettify_table(acc):
    """Prettify dataframe for analysis and presentation
    """
    # Replace label wordings
    acc['train_dataset'].replace({
        'FEVER': 'Fine-tuned on FEVER',
        'PUBHEALTH': 'Fine-tuned on PUBHEALTH',
        # 'CLIMATE': 'Fine-tuned on CLIMATE-FEVER',  #change back
        'CLIMATE-FEVER': 'Fine-tuned on CLIMATE-FEVER',
    }, inplace=True)

    acc['test_dataset'].replace({
        'fever': 'FEVER',
        'pubhealth': 'PUBHEALTH',
        'climate': 'CLIMATE-FEVER'
    }, inplace=True)

    # pivot by model for better presentation
    acc_pivot = acc.pivot(index=['train_dataset', 'test_dataset'], columns='model', values='correctly_classified')
    acc_pivot = acc_pivot[['BERT', 'RoBERTa', 'ALBERT', 'SciBERT', 'BioBERT']].\
        reindex(['Fine-tuned on FEVER', 'Fine-tuned on PUBHEALTH', 'Fine-tuned on CLIMATE-FEVER'], level=0).\
        reindex(['FEVER', 'PUBHEALTH', 'CLIMATE-FEVER'], level=1)

    return acc_pivot



    