#Todo: combine with evaluate_model.py
#Code used to return samples for analysis by labels

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import pandas as pd

def _get_predictions(model, ds, device):
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


def get_predictions(model_checkpoint, ds_test):
    """Evaluate accuracy of saved model on test datasets
    
    Args:
        mdoel_checkpoint (string): path to best model,
        ds_test (DatasetDict): huggingface dataset for fever_test, pubhealth_test, climate_test,
    """

    #===================================================
    # Load Model
    #===================================================
    num_labels = 3 
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Model loaded into {device}")
    model.to(device)

    #===================================================
    # Tokenize dataset
    #===================================================
    print(f"Tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def preprocess_function(samples):
        return tokenizer(samples['claim'], samples['evidence'], 
                         padding=True,
                         truncation='only_second', 
                         max_length=512)

    encoded_ds = ds_test.map(preprocess_function, batched=True)

    # format tokens to fit huggingface language model formats
    encoded_ds = encoded_ds.remove_columns(["claim", "evidence"])
    encoded_ds = encoded_ds.rename_column("label", "labels")
    encoded_ds.set_format("torch")

    #===================================================
    # Evaluate
    #===================================================    
    results = dict()
    for ds_name in encoded_ds.keys():
        print(f"Evaluating {ds_name}")
        eval_ds = DataLoader(encoded_ds[ds_name], batch_size=8)
        predictions = _get_predictions(model, eval_ds, device)

        # combine into dataframe for further analysis
        df = pd.DataFrame(ds_test[ds_name])
        df['pred'] = predictions
        df['correctly_classified'] = df['label'] == df['pred']
        results[ds_name] = df.copy()
        
    return results


