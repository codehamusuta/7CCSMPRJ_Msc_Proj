# -*- coding: utf-8 -*-
# -----------------------
# Author : Wen Ting, Choi
# Date : 2023-08-09 
# -----------------------

"""Evaluate finetuned models on test datasets

Predictions will be combined into a single dataframe and saved to predictions_fp. 
Before running, ensure config.yaml is set to the correct directories. 
"""

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import pandas as pd

from experiments.finetuning import tokenize_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _get_predictions(model, ds, device=device):
    """
    Args:
        model (pytorch model): model to evaluate
        ds (torch.DataLoader): dataset to evaluate on loaded into pytorch DataLoader obj
        device (torch.device): GPU / CPU
    """        
    model.eval()
    predictions = []
    for batch in ds:
        batch = {k: v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions = predictions + preds.tolist()
        
    predictions = np.array(predictions)
    return predictions


def get_predictions(model_checkpoint, ds_test, debug=False):
    """Evaluate accuracy of saved model on test datasets

    Args:
        model_checkpoint (string): path to fine-tuned model,
        ds_test (DatasetDict): huggingface dataset for fever_test, pubhealth_test, climate_test,
        debug (bool, optional): print debugging info. Defaults to False.

    Returns:
        dict(pd.DataFrame): return dictionary of dataframes with predictions for each test dataset
    """    
    #===================================================
    # Load Model
    #===================================================
    num_labels = 3 
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    model.to(device)
    if debug:
        print(f"Model loaded into {device}")
    
    #===================================================
    # Tokenize dataset
    #===================================================
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    encoded_ds = tokenize_data(ds_test, tokenizer)

    # format tokens to fit huggingface language model formats
    encoded_ds = encoded_ds.remove_columns(["claim", "evidence"])
    encoded_ds = encoded_ds.rename_column("label", "labels")
    encoded_ds.set_format("torch")

    if debug:
        print(f"Datasets Tokenized")

    #===================================================
    # Evaluate
    #===================================================    
    results = dict()
    for ds_name in encoded_ds.keys(): #fever, pubhealth, climate
        if debug:
            print(f"Evaluating {ds_name}")
        
        eval_ds = DataLoader(encoded_ds[ds_name], batch_size=8)
        predictions = _get_predictions(model, eval_ds)

        # combine into dataframe for further analysis
        df = pd.DataFrame(ds_test[ds_name])
        df['pred'] = predictions
        df['correctly_classified'] = df['label'] == df['pred']
        results[ds_name] = df.copy()
        
    return results



if __name__ == "__main__":
    import os

    from experiments.dataloader import load_datasets
    from experiments.utils import load_config
    from experiments.finetuning import MODELS, DATASETS
    
    config = load_config()

    processed_data_dir = config["processed_data_dir"]
    models_dir = config["evaluation"]["models_dir"]
    output_fp = config["evaluation"]["predictions_fp"]

    experiments = [
        {"model": model_name, 
         "train_dataset": ds_name, 
         "best_model_path": os.path.join(models_dir, f"{model_name}_{ds_name}", "best_model")} 
        for model_name in MODELS
        for ds_name in DATASETS
    ]
    
    # load data
    ds_test = load_datasets(processed_data_dir)['TEST']
    
    # run all experiments
    results = []
    for experiment in experiments:
        print(f"Evaluating {experiment['model']} finetuned on {experiment['train_dataset']}")

        model_path = experiment['best_model_path']
        predictions = get_predictions(model_path, ds_test)
        for test_dataset, _df in predictions.items():
            d = experiment.copy()
            d['test_dataset'] = test_dataset
            d['predictions'] = _df.copy() 
            results.append(d)


    # combine results into a single dataset for analysis
    results_df = []
    for d in results:
        _df = d['predictions'].copy()
        _df['model'] = d['model']
        _df['train_dataset'] = d['train_dataset']
        _df['test_dataset'] = d['test_dataset']
        results_df.append(_df)

    results_df = pd.concat(results_df)

    results_df.to_picke(output_fp)
    print(f"Predictions saved to {output_fp}")
