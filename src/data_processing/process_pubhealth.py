# -*- coding: utf-8 -*-
# -----------------------
# Author : Wen Ting, Choi
# Date : 2023-08-09 
# -----------------------
"""Data processing script for PUBHEALTH data
"""
import os
from operator import itemgetter

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from data_processing.utils import load_config


def load_data(raw_data_dir):
    train_df = pd.read_csv(os.path.join(raw_data_dir, 'train.tsv'), sep="\t")
    dev_df = pd.read_csv(os.path.join(raw_data_dir, 'dev.tsv'), sep="\t")
    test_df = pd.read_csv(os.path.join(raw_data_dir, 'test.tsv'), sep="\t")

    return train_df, dev_df, test_df

def drop_missing_data(df):
    """drop data points w/o main text"""    
    _df = df[df['main_text'].notnull()].copy()
    return _df

def drop_labels(df):
    """filter by labels"""
    _df = df[df['label'].isin(['true', 'false', 'unproven'])].copy()
    return _df

def standardize_labels(df):
    """standardize label names"""
    _df = df.copy()
    _df['label'].replace({
        "true": "SUPPORTS", 
        "false": "REFUTES", 
        "unproven": "NOT ENOUGH INFO"}, inplace=True)
    return _df
    
def select_evidence_sentences(df, k=5):
    """select top k evidence sentences based on sentence transformer model"""
    corpus = df.copy()
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    model.to("cuda")
    corpus['evidence'] = np.empty([len(corpus),], dtype=str)

    for index, row in corpus.iterrows():
        claim = row['claim']
        sentences = [claim] + [sent for sent in sent_tokenize(row['main_text'])]
        
        sentence_embeddings = model.encode(sentences)
        claim_embedding = sentence_embeddings[0]
        sentence_embeddings = sentence_embeddings[1:]
        cosine_similarity_emb = {}
        
        for sent, embedding in zip(sentences[1:], sentence_embeddings):
            cosine_similarity_emb[sent] = np.linalg.norm(
                cosine_similarity([claim_embedding, embedding])
            )
            
        top_k = dict(sorted(cosine_similarity_emb.items(),
                            key=itemgetter(1))[:k])
        
        # concatenate top k sentence into a single text as evidence
        corpus.at[index, 'evidence'] = " ".join([key for key in top_k.keys()])

    return corpus

def extract_evidence(df):
    """extract top 5 sentences as evidence from main_text"""
    return select_evidence_sentences(df)

def standardize_fieldnames(df):
    #only keep these 3 columns
    columns = ['claim', 'label', 'evidence']
    _df = df[columns].copy()
    return _df

def transform_df(df):
    pipeline = [
        drop_missing_data,
        drop_labels,
        standardize_labels,
        extract_evidence,
        standardize_fieldnames
    ]
    for func in pipeline:
        df = func(df)
    return df

def save_processed_data(train_df, dev_df, test_df, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_df.to_json(
        os.path.join(output_dir, "pubhealth_train.jsonl"),
        orient="records"
    )

    dev_df.to_json(
        os.path.join(output_dir, "pubhealth_dev.jsonl"),
        orient="records"
    )

    test_df.to_json(
        os.path.join(output_dir, "pubhealth_test.jsonl"),
        orient="records"
    )


def process_pubhealth():
    """Main function to load and process PUBHEALTH data
    """
    config = load_config()
    raw_data_dir = config['raw_data_dirs']['pubhealth']
    processsed_data_dir = config["processed_data_dir"]

    # load data
    train_df, dev_df, test_df = load_data(raw_data_dir)

    # transform data
    train_df = transform_df(train_df)
    dev_df = transform_df(dev_df)
    test_df = transform_df(test_df)

    # save processed data
    save_processed_data(train_df, dev_df, test_df, processsed_data_dir)

if __name__ == "__main__":
    process_pubhealth()
    print("Done")