models = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "ALBERT": "albert-base-v1",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "BioBERT": "dmis-lab/biobert-base-cased-v1.2"
}
datasets = ["FEVER", "PUBHEALTH", "CLIMATE"]

import os
import shutil

import numpy as np
import torch
from torch import nn
import evaluate 
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.utils import class_weight

def fine_tune_model(pretrained_model, model_dir, ds, batch_size=8, num_epochs=10, save_total_limit=2):

    # if model_path exist, skip
    best_model_path = os.path.join(model_dir, 'best_model')
    if os.path.exists(best_model_path):
        print(f"Found {best_model_path}. Skipping to next model")
        return 

    try:
        #=======================================================
        # Tokenize data
        #=======================================================
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        def preprocess_function(samples):
            return tokenizer(samples['claim'], samples['evidence'], 
                             padding=True,
                             truncation='only_second', 
                             max_length=512)
    
        encoded_ds = ds.map(preprocess_function, batched=True)
    
        #=======================================================
        # Load pre-trained model
        #=======================================================
        num_labels = 3
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
    
        #=======================================================
        # Setup trainer
        #=======================================================
        metric = evaluate.load("accuracy")
    
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references = labels)
        
        args = TrainingArguments(
            model_dir,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            num_train_epochs = num_epochs,
            load_best_model_at_end = True,
            logging_strategy = "epoch",
            save_total_limit = save_total_limit,
        )  
    
        class_weights = class_weight.compute_class_weight(
            "balanced", 
            classes=np.array([0,1,2]), 
            y=encoded_ds["train"]["label"]
        )
    
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss
    
        trainer = CustomTrainer(
            model, 
            args,
            train_dataset = encoded_ds["train"],
            eval_dataset = encoded_ds["validation"],
            tokenizer = tokenizer, 
            compute_metrics = compute_metrics
        )
        
        #=======================================================
        # Train model
        #=======================================================
        torch.cuda.empty_cache()
        trainer.train()
        trainer.save_model(best_model_path)
        print(f"Best model saved to {best_model_path}")
    except Exception as e:
        print(f"Error running {model_dir} with error {e}")
        shutil.rmtree(model_dir)
        print(f"Skipping to next experiment.")

    return
        


    
