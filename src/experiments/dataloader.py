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

def load_data(data_dir, ds_name):
    if ds_name not in ["fever", "pubhealth", "climate"]:
        raise ValueError("Unrecognised dataset name")
    
    train_ds = read_json(os.path.join(data_dir, f'{ds_name}_train.jsonl'))
    dev_ds = read_json(os.path.join(data_dir, f'{ds_name}_dev.jsonl'))
    test_ds = read_json(os.path.join(data_dir, f'{ds_name}_test.jsonl'))

    return train_ds, dev_ds, test_ds
    
def load_datasets(data_dir):
    """load processed data into HuggingFace dataset objects 

    Args:
        data_dir (str): folder path to processed data
    """    

    fever_train_ds, fever_dev_ds, fever_test_ds = load_data(data_dir, "fever")
    pubhealth_train_ds, pubhealth_dev_ds, pubhealth_test_ds = load_data(data_dir, "pubhealth")
    climate_train_ds, climate_dev_ds, climate_test_ds = load_data(data_dir, "climate")
    
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
    ds_fever['train'] = Dataset.from_list(fever_train_ds, features=features)
    ds_fever['validation'] = Dataset.from_list(fever_dev_ds, features=features)

    #for fine-tuning on PUBHEALTH
    ds_pubhealth = DatasetDict()
    ds_pubhealth['train'] = Dataset.from_list(pubhealth_train_ds, features=features)
    ds_pubhealth['validation'] = Dataset.from_list(pubhealth_dev_ds, features=features)

    #for fine-tuning on CLIMATE
    ds_climate = DatasetDict()
    ds_climate['train'] = Dataset.from_list(climate_train_ds, features=features)
    ds_climate['validation'] = Dataset.from_list(climate_dev_ds, features=features)

    #for evaluation 
    ds_test = DatasetDict()
    ds_test['fever'] = Dataset.from_list(fever_test_ds, features=features)
    ds_test['pubhealth'] = Dataset.from_list(pubhealth_test_ds, features=features)
    ds_test['climate'] = Dataset.from_list(climate_test_ds, features=features)

    # return ds_fever, ds_pubhealth, ds_climate, ds_test
    return {
        "FEVER": ds_fever,
        "PUBHEALTH": ds_pubhealth,
        "CLIMATE": ds_climate,
        "TEST": ds_test 
    }
    

