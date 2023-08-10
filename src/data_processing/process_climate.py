import os
import json

from sklearn.model_selection import train_test_split

from .utils import load_config


# filter labels
def drop_labels(sample):
    return sample['claim_label'] in ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO']

# standardize labels
def standardize_labels(sample):
    label = sample['claim_label']
    if label == "NOT_ENOUGH_INFO":
        sample["claim_label"] = "NOT ENOUGH INFO"
    return sample

# extract evidences
def extract_evidence(sample):
    evidences = sample["evidences"]

    # concatenate evidence 
    sample["evidences"] = " ".join([e["evidence"] for e in evidences]) #evidences is a list of dictionary 
    return sample

# standardize fieldnames
def standardize_fieldnames(sample):
    d = dict()
    d['claim'] = sample['claim']
    d['label'] = sample['claim_label']
    d['evidence'] = sample['evidences']
    return d
    
# split dataset 
def split_dataset(data, dev_size=200, test_size=200, random_state = 392):
    # Split climate_ds into train & test
    train, test = train_test_split(
        data, 
        test_size = test_size, 
        random_state = random_state, 
        stratify=[d['label'] for d in data]
    )
    train, dev = train_test_split(
        train, 
        test_size = dev_size, 
        random_state = random_state, 
        stratify=[d['label'] for d in train]
    )

    return train, dev, test

def save_processed_data(train, dev, test, output_dir, dataset_name): #fever, pubhealth, climate
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, f"{dataset_name}_train.jsonl"), 'w') as f:
        json.dump(train, f)

    with open(os.path.join(output_dir, f"{dataset_name}_dev.jsonl"), 'w') as f:
        json.dump(dev, f)

    with open(os.path.join(output_dir, f"{dataset_name}_test.jsonl"), 'w') as f:
        json.dump(test, f)
    

def process_climate():
    """Main function to load and process CLIMATE-FEVER data
    """
    config = load_config()
    raw_data_dir = config['raw_data_dirs']['climate']
    processsed_data_dir = config["processed_data_dir"]

    # load data
    with open(os.path.join(raw_data_dir, 'climate-fever-dataset-r1.jsonl'), 'r') as f:
        data = [json.loads(item) for item in list(f)]

    # transform data
    data = list(map(standardize_fieldnames, 
                map(extract_evidence, 
                map(standardize_labels, 
                filter(drop_labels, data)))))
    
    # split dataset
    train, dev, test = split_dataset(data)

    # save processed data
    save_processed_data(train, dev, test, processsed_data_dir, 'climate')
    


if __name__ == "__main__":
    process_climate()