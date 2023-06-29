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


        
        