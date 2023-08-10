# -*- coding: utf-8 -*-
# -----------------------
# Author : Wen Ting, Choi
# Date : 2023-08-09 
# -----------------------
"""Finetuning of models for all experiments

Best models will be saved to <models_dir>/<model_name>_<dataset_name>/best_model. 
Setup necessary configurations in config.yaml before running.
"""
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

#Define Constants 
DATASETS = ["FEVER", "PUBHEALTH", "CLIMATE"]
MODELS =  ["BERT", "RoBERTa", "ALBERT", "SciBERT", "BioBERT"]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def tokenize_data(ds, tokenizer):
    """Encode data using tokenizer

    Args:
        ds (DatasetDict): dataset in HuggingFace Dataset or DatasetDict format
        tokenizer (object): HuggingFace transformer tokenizer 

    Returns:
        DatasetDict: encoded dataset
    """
    def preprocess_function(samples):
        return tokenizer(samples['claim'], samples['evidence'], 
                            padding=True,
                            truncation='only_second', 
                            max_length=512)

    encoded_ds = ds.map(preprocess_function, batched=True)
    return encoded_ds


class NormalCELoss:
    @staticmethod
    def get_trainer(self, *args, **kwargs):
        return Trainer
    
class WeightedCELoss:
    @staticmethod
    def get_trainer(self, ds, device=device):
        #class weights needs to be computed on the fly because different training 
        #datasets will have different class weights
        class_weights = class_weight.compute_class_weight(
            "balanced", 
            classes=np.array([0,1,2]), 
            y=ds["train"]["label"]
        )
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
            
        return CustomTrainer


def gen_compute_metrics_fn(metric="accuracy"):
    """Generate function to compute metrics at evaluation as required by 
    trainer API
    """
    metric = evaluate.load(metric)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references = labels)
    return compute_metrics

class Experiment:
    """Context to run experiments
    """    
    def __init__(
            self, 
            TrainerContext = NormalCELoss, 
            best_model_fn = 'best_model',
            eval_metric = "accuracy"
        ):
        self._trainer_context = TrainerContext
        self._best_model_fn = best_model_fn
        self._compute_metrics = gen_compute_metrics_fn(eval_metric)
        
    def finetune_model(self, pretrained_model, 
                       model_dir, ds, batch_size=8, 
                       num_epochs=5, save_total_limit=2):
        """Finetune pre-trained model on a training dataset

        Args:
            pretrained_model (str): path to HuggingFace pre-trained model
            model_dir (str): path to saved model checkpoints and best model
            ds (DatasetDict): training dataset as returned from load_datasets method
            batch_size (int, optional): Defaults to 8.
            num_epochs (int, optional): Defaults to 5.
            save_total_limit (int, optional): number of model checkpoints to save during training. 
                Defaults to 2 (best_model and last checkpoint).
        """        
        
        #if model_path exist, skip
        best_model_path = os.path.join(model_dir, self._best_model_fn)
        if os.path.exists(best_model_path):
            print(f"Found {best_model_path}. Skipping to next model")
            return 
        
        try:
            #=======================================================
            # Tokenize data
            #=======================================================
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            encoded_ds = tokenize_data(ds, tokenizer)
        
            #=======================================================
            # Load pre-trained model
            #=======================================================
            num_labels = 3
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
        
            #=======================================================
            # Setup trainer
            #=======================================================        
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

            # Need to create class dynamically for weighted loss version
            TrainerClass = self._trainer_context.get_trainer(ds) 

            trainer = TrainerClass(
                model, 
                args,
                train_dataset = encoded_ds["train"],
                eval_dataset = encoded_ds["validation"],
                tokenizer = tokenizer, 
                compute_metrics = self._compute_metrics
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
    

if __name__ == "__main__":

    import os

    from .utils import load_config
    from .dataloader import load_datasets

    config = load_config()

    #===========================================================================
    # Get configs
    #===========================================================================
    pretrained_models = config["pretrained_models"]  # dict[model_name]= pretrained_model_fp
    models_dir = config["training"]["models_dir"]
    
    def gen_experiment_params():
        for model_name, pretrained_model in pretrained_models.items():
            for dataset_name in DATASETS:
                model_dir = os.path.join(models_dir, f"{model_name}_{dataset_name}")
                yield (model_name, dataset_name, pretrained_model, model_dir)
    
    #===========================================================================
    # Setup experiments
    #===========================================================================
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    experiments = list(gen_experiment_params())

    datasets = load_datasets(config["processed_data_dir"])

    experiment_type = config["training"]["experiment_type"]
    if experiment_type == "normal_loss":
        context = Experiment(TrainerContext = NormalCELoss)
    elif experiment_type == "weighted_loss":
        context = Experiment(TrainerContext = WeightedCELoss)

    #===========================================================================
    # Run experiments
    #===========================================================================
    for model_name, dataset_name, pretrained_model, model_dir in experiments:
        train_ds = datasets[dataset_name]
        
        print(f"Finetuning {model_name} with {train_ds}")
        context.finetune_model(pretrained_model, model_dir, train_ds, batch_size=8, 
                               num_epochs=5, save_total_limit=2)