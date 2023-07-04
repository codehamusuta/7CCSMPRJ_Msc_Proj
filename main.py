"""
ETL functions
"""
import json
import os

def read_json(fp):
    with open(fp, "r", encoding="utf-8") as f:
        data = []
        for line in f.readlines():
            data.append(json.loads(line.strip()))
        return data

class FeverLoader:
    @staticmethod
    def process_fever(sample):
        #concatenate evidence_text
        obj = {}
        obj['claim'] = sample['claim']
        obj['label'] = sample['label']
        obj['evidence'] = " ".join(sample['evidence_text'])
        return obj

    @staticmethod
    def load(fp):
        """
        Args:
            fp (string): folder path with preprocessed fever data
        """
        #load the version with random sampling
        fever_test_ds = read_json(os.path.join(fp, 'test_preprocessed.ns.rand.jsonl'))
        fever_dev_ds = read_json(os.path.join(fp, 'dev_preprocessed.ns.rand.jsonl'))
        fever_train_ds = read_json(os.path.join(fp, 'train_preprocessed.ns.rand.jsonl'))
    
        fever_test_ds = list(map(FeverLoader.process_fever, fever_test_ds))
        fever_dev_ds = list(map(FeverLoader.process_fever, fever_dev_ds))
        fever_train_ds = list(map(FeverLoader.process_fever, fever_train_ds))
    
        return fever_train_ds, fever_dev_ds, fever_test_ds

class PubhealthLoader:
    @staticmethod
    def filter_pubhealth(sample):
        return sample['label'] in ['true', 'false', 'unproven']
    
    @staticmethod
    def process_pubhealth(sample):
        obj = {}
        obj['claim'] = sample['claim']
        obj['evidence'] = " ".join(sample['top_k'])
        
        # modify label
        label = sample["label"]
        if label == 'true':
            obj["label"] = "SUPPORTS"
        elif label == 'false':
            obj["label"] = "REFUTES"
        else:
            obj["label"] = "NOT ENOUGH INFO"
        return obj
    
    @staticmethod
    def load(fp):
        """
        Args:
            fp (string): folder path with preprocessed fever data
        """
        pubhealth_train_ds = read_json(os.path.join(fp, 'train.jsonl'))
        pubhealth_dev_ds = read_json(os.path.join(fp, 'dev.jsonl'))
        pubhealth_test_ds = read_json(os.path.join(fp, 'test.jsonl'))
    
        pubhealth_train_ds = list(filter(PubhealthLoader.filter_pubhealth, pubhealth_train_ds))
        pubhealth_dev_ds = list(filter(PubhealthLoader.filter_pubhealth, pubhealth_dev_ds))
        pubhealth_test_ds = list(filter(PubhealthLoader.filter_pubhealth, pubhealth_test_ds))
    
        pubhealth_train_ds = list(map(PubhealthLoader.process_pubhealth, pubhealth_train_ds))
        pubhealth_dev_ds = list(map(PubhealthLoader.process_pubhealth, pubhealth_dev_ds))
        pubhealth_test_ds = list(map(PubhealthLoader.process_pubhealth, pubhealth_test_ds))
    
        return pubhealth_train_ds, pubhealth_dev_ds, pubhealth_test_ds

class ClimateFeverLoader:
    @staticmethod
    def process_climate(sample):
        obj = {}
        obj['claim'] = sample['claim']
    
        #modify label
        label = sample['claim_label']
        if label == "NOT_ENOUGH_INFO":
            label = "NOT ENOUGH INFO"
        obj['label'] = label
    
        #concatenate evidence
        obj['evidence'] = " ".join([e['evidence'] for e in sample['evidences']])
    
        return obj

    @staticmethod
    def load(fp):
        """
        Args:
            fp (string): folder path with preprocessed fever data
        """
        climate_ds = read_json(os.path.join(fp, 'climate-fever.jsonl'))
        climate_ds = list(map(ClimateFeverLoader.process_climate, climate_ds))

        return climate_ds

"""
Prepare dataset for experiments
"""
from datasets import Dataset, DatasetDict, ClassLabel, Value, Features
from sklearn.model_selection import train_test_split

def load_datasets(fever_dir, pubhealth_dir, climate_dir):
    """
    Load and prepare datasets for experiments
    """
    #===================================================
    # Load preprocessed datasets
    #===================================================
    fever_train_ds, fever_dev_ds, fever_test_ds = FeverLoader.load(fever_dir)
    pubhealth_train_ds, pubhealth_dev_ds, pubhealth_test_ds = PubhealthLoader.load(pubhealth_dir)
    climate_ds = ClimateFeverLoader.load(climate_dir)

    # Split climate_ds into train & test
    climate_train_ds, climate_test_ds = train_test_split(
        climate_ds, 
        test_size=200, 
        random_state=392, 
        stratify=[d['label'] for d in climate_ds]
    )
    climate_train_ds, climate_dev_ds = train_test_split(
        climate_train_ds, 
        test_size=200, 
        random_state=392, 
        stratify=[d['label'] for d in climate_train_ds]
    )

    #===================================================
    # Setup huggingface dataset objects
    #===================================================
    features = Features({
        "claim": Value("string"), 
        "evidence": Value("string"),
        "label": ClassLabel(num_classes=3, names=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
    })
    
    # train on fever 
    ds1 = DatasetDict()
    ds1['train'] = Dataset.from_list(fever_train_ds, features=features)
    ds1['validation'] = Dataset.from_list(fever_dev_ds, features=features)
    ds1['fever_test'] = Dataset.from_list(fever_test_ds, features=features)
    ds1['pubhealth_test'] = Dataset.from_list(pubhealth_test_ds, features=features)
    ds1['climate_test']  = Dataset.from_list(climate_test_ds, features=features)

    # train on pubhealth
    ds2 = DatasetDict()
    ds2['train'] = Dataset.from_list(pubhealth_train_ds, features=features)
    ds2['validation'] = Dataset.from_list(pubhealth_dev_ds, features=features)
    ds2['fever_test'] = Dataset.from_list(fever_test_ds, features=features)
    ds2['pubhealth_test'] = Dataset.from_list(pubhealth_test_ds, features=features)
    ds2['climate_test']  = Dataset.from_list(climate_test_ds, features=features)

    # train on climate
    ds3 = DatasetDict()
    ds3['train'] = Dataset.from_list(climate_train_ds, features=features)
    ds3['validation'] = Dataset.from_list(climate_dev_ds, features=features)
    ds3['fever_test'] = Dataset.from_list(fever_test_ds, features=features)
    ds3['pubhealth_test'] = Dataset.from_list(pubhealth_test_ds, features=features)
    ds3['climate_test']  = Dataset.from_list(climate_test_ds, features=features)

    # test datasets
    ds_test = DatasetDict()
    ds_test['fever'] = Dataset.from_list(fever_test_ds, features=features)
    ds_test['pubhealth'] = Dataset.from_list(pubhealth_test_ds, features=features)
    ds_test['climate'] = Dataset.from_list(climate_test_ds, features=features)

    return ds1, ds2, ds3, ds_test


        
        