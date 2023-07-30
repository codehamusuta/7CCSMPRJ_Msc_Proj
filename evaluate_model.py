from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import evaluate

def _evaluate(model, ds, device, metric=None):
    """
    Args:
        model (pytorch model): model to evaluate
        ds (torch.DataLoader): dataset to evaluate on loaded into pytorch DataLoader obj
        device (torch.device): GPU / CPU
        metric (string): evaluation metrics to use. Defaults to accuracy.
    """
    if metric is None:
        raise ValueError("Please include a metric to evaluate")
        
    if isinstance(metric, list):
        metric = evaluate.combine(metric)
    elif isinstance(metric, str):
        metric = evaluate.load(metric)
    else:
        raise ValueError("Metric is of invalid type")

    model.eval()
    for batch in ds:
        batch = {k: v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    return metric.compute()

def evaluate_model(model_checkpoint, ds_test, metric=None, verbose=False):
    """Evaluate accuracy of saved model on test datasets
    
    Args:
        mdoel_checkpoint (string): path to best model,
        ds_test (DatasetDict): huggingface dataset for fever_test, pubhealth_test, climate_test,
        metric (string): evaluation metrics to use. Defaults to accuracy.
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
        r = _evaluate(model, eval_ds, device, metric)
        if verbose:
            print(f"{ds_name} :: {r}")
        results[ds_name] = r
        
    return results
        
    