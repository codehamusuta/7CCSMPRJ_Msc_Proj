# -*- coding: utf-8 -*-
# -----------------------
# Author : Wen Ting, Choi
# Date : 2023-08-09 
# -----------------------

"""Prepare datasets for experiments

Load processed data files from a folder containing those files.
Files should be name <dataset>_training.jsonl, <dataset>_dev.jsonl, and 
<dataset>_test.jsonl
"""

import json
import os

from datasets import Dataset, DatasetDict, ClassLabel, Value, Features

def read_json(fp):
    with open(fp, "r", encoding="utf-8") as f:
        data = []
        for line in f.readlines():
            data.append(json.loads(line.strip()))
        return data


def get_fps(data_dir, ds_name):
    train_fp = os.path.join(data_dir, f'{ds_name}_train.jsonl')
    dev_fp = os.path.join(data_dir, f'{ds_name}_dev.jsonl')
    test_fp = os.path.join(data_dir, f'{ds_name}_test.jsonl')

    return train_fp, dev_fp, test_fp

    
def load_datasets(data_dir):
    """load processed data into HuggingFace dataset objects 

    Args:
        data_dir (str): folder path to processed data
    """    

    fever_train_fp, fever_dev_fp, fever_test_fp = get_fps(data_dir, "fever")
    pubhealth_train_fp, pubhealth_dev_fp, pubhealth_test_fp = get_fps(data_dir, "pubhealth")
    climate_train_fp, climate_dev_fp, climate_test_fp = get_fps(data_dir, "climate")
    
    #===================================================
    # Setup huggingface dataset objects
    #===================================================
    features = Features({
        "claim": Value("string"), 
        "evidence": Value("string"),
        "label": ClassLabel(num_classes=3, names=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
    })

    #for fine-tuning on FEVER
    ds_fever = DatasetDict()
    ds_fever['train'] = Dataset.from_json(fever_train_fp, features=features)
    ds_fever['validation'] = Dataset.from_json(fever_dev_fp, features=features)

    #for fine-tuning on PUBHEALTH
    ds_pubhealth = DatasetDict()
    ds_pubhealth['train'] = Dataset.from_json(pubhealth_train_fp, features=features)
    ds_pubhealth['validation'] = Dataset.from_json(pubhealth_dev_fp, features=features)

    #for fine-tuning on CLIMATE
    ds_climate = DatasetDict()
    ds_climate['train'] = Dataset.from_json(climate_train_fp, features=features)
    ds_climate['validation'] = Dataset.from_json(climate_dev_fp, features=features)

    #for evaluation 
    ds_test = DatasetDict()
    ds_test['fever'] = Dataset.from_json(fever_test_fp, features=features)
    ds_test['pubhealth'] = Dataset.from_json(pubhealth_test_fp, features=features)
    ds_test['climate'] = Dataset.from_json(climate_test_fp, features=features)

    # return ds_fever, ds_pubhealth, ds_climate, ds_test
    return {
        "FEVER": ds_fever,
        "PUBHEALTH": ds_pubhealth,
        "CLIMATE": ds_climate,
        "TEST": ds_test 
    }
    

