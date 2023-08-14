## About

This repository contains source codes including data processing scripts, model training 
and evaluation scripts for the MSc Project titled `Robustness of Fact Checking Models`.

## Declaration of work

I verify that I am the sole author of the programmes contained in this archive, except where explicitly stated to the contrary.

Wen Ting, Choi
14th August 2023

## List of files
- data_processing/process_climate.py
- data_processing/process_fever.py
- data_processing/process_pubhealth.py
- data_processing/utils.py
- experiments/dataloader.py
- experiments/evaluate.py
- experiments/finetuning.py
- experiments/utils.py
- config.yaml

## Instructions

1. Download respective raw data into raw data folder
2. Run the data processing scripts to extract and transform the raw data
    ```bash
    #before running, 
    #cd into root folder of source code
    #configure config.yaml to point to the appropriate folder

    python3 -m data_processing.process_climate
    python3 -m data_processing.process_fever
    python3 -m data_processing.process_pubhealth
    ```
3. Run finetuning.py to fine-tune the models
    ```bash
    #cd into root folder of source code
    #configure config.yaml to the correct folders and settings
    python3 -m experiiments.finetuning
    ```
4. Run evaluate to get predictions for all test datasets
    ```bash
    #cd into root folder of source code
    #configure config.yaml to the correct folders and settings
    python3 -m experiments.evaluate
    ```