# -*- coding: utf-8 -*-
# -----------------------
# Author : Wen Ting, Choi
# Date : 2023-08-09 
# -----------------------
"""Data processing script for FEVER data
"""
import os
import json
import unicodedata
import sqlite3

import pandas as pd

from data_processing.utils import load_config, SimpleRandom

def read_json(json_fp):
    """Read in json records and return dataframe
    """
    with open(json_fp, 'r') as f:
        data = [json.loads(item) for item in list(f)]
        df = pd.DataFrame.from_records(data)
    return df

def load_data(raw_data_dir):
    train_df = read_json(os.path.join(raw_data_dir, 'train.ns.rand.jsonl'))
    dev_df = read_json(os.path.join(raw_data_dir, 'dev.ns.rand.jsonl'))
    test_df = read_json(os.path.join(raw_data_dir, 'test.ns.rand.jsonl'))

    return train_df, dev_df, test_df

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

class DocDB():
    """Sqlite interactor. Use to interact with wikidb from fever.

    Implements get_doc_text(doc_id).
    """
    def __init__(self, db_path):
        self.path = db_path 
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
    
    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
    
    def get_non_empty_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents WHERE length(trim(text)) > 0")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

class EvidenceExtractor:
    """Extract evidence from wikipedia db
    """    
    def __init__(self, db):
        self._db = db

    def _get_doc_line(self, doc, line):
        lines = self._db.get_doc_lines(doc)
        if line > -1:
            return lines.split("\n")[line].split("\t")[1] #get specific line in wiki page
        else:
            non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
            return non_empty_lines[SimpleRandom.get_instance().next_rand(0,len(non_empty_lines)-1)] #get random sent from doc

    def extract_evidence(self, evidence):
        try:
            pages = []
            for evidence_group in evidence:
                pages.extend([(ev[2],ev[3]) for ev in evidence_group])
            
            lines = list(set([self._get_doc_line(d[0],d[1]) for d in pages]))
            
            # concatenate evidences into a single text 
            evidence = " ".join(lines) 
            return evidence
        
        except Exception as e:
            print(evidence)
            print(e)
            raise

def extract_evidence(df, evidence_extractor):
    _df = df.copy()
    _df['evidence'] = _df['evidence'].apply(
        lambda evidence: evidence_extractor.extract_evidence(evidence)
    )
    return _df

def standardize_fieldnames(df):
    #only keep these 3 columns
    columns = ['claim', 'label', 'evidence']
    _df = df[columns].copy()
    return _df

def save_processed_data(train_df, dev_df, test_df, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_df.to_json(
        os.path.join(output_dir, "fever_train.jsonl"),
        orient="records"
    )
    dev_df.to_json(
        os.path.join(output_dir, "fever_dev.jsonl"),
        orient="records"
    )
    test_df.to_json(
        os.path.join(output_dir, "fever_test.jsonl"),
        orient="records"
    )

def process_fever():
    """Main function to load and process FEVER data
    """
    config = load_config()
    raw_data_dir = config['raw_data_dirs']['fever']
    processsed_data_dir = config["processed_data_dir"]

    # load data
    train_df, dev_df, test_df = load_data(raw_data_dir)

    # setup connection to wikipedia db (sqlite)
    db = DocDB(os.path.join(raw_data_dir, 'fever.db'))
    evidence_extractor = EvidenceExtractor(db)

    # transform data
    train_df = standardize_fieldnames(
        extract_evidence(train_df, evidence_extractor))
    dev_df = standardize_fieldnames(
        extract_evidence(dev_df, evidence_extractor))
    test_df = standardize_fieldnames(
        extract_evidence(test_df, evidence_extractor))

    # save processed data
    save_processed_data(train_df, dev_df, test_df, processsed_data_dir)



if __name__ == "__main__":
    process_fever()
    print("Done")