import os
import json
import copy
import requests
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import importlib

import open_alex_utils

importlib.reload(open_alex_utils)

from open_alex_utils import get_pos_citation_dataset, \
    get_neg_citation_dataset, extract_bow_feats, get_bow_inputs_labels


pos_path = "../data/pubmed_articles_pos_human.csv"
    
neg_path = "../data/pubmed_articles_neg_human.csv"

ids_file = '/home/k2141255/study_design_classifier/data/cleaned_pos_neg_train_val_test_dataset_ids.pkl'
with open(ids_file, 'rb') as f:
    prop_typ_id_dict= pkl.load(f)
    
pmids_dict = {}
PMIDS = '../data/pmids.pkl'

for study_design in tqdm(prop_typ_id_dict):
    pos_ids = prop_typ_id_dict[study_design]['pos']
    neg_ids = prop_typ_id_dict[study_design]['neg']
    
    with open(pos_path, 'r') as f:
        pos_data = pd.read_csv(f)
    
    pos_train_pmids = [pos_data.loc[pos_id]['pmid'] for pos_id in pos_ids['train']]
    pos_val_pmids = [pos_data.loc[pos_id]['pmid'] for pos_id in pos_ids['val']]
    pos_test_pmids = [pos_data.loc[pos_id]['pmid'] for pos_id in pos_ids['test']]
    del pos_data
    
    with open(neg_path, 'r') as f:
        neg_data = pd.read_csv(f)
    
    neg_train_pmids = [neg_data.loc[neg_id]['pmid'] for neg_id in neg_ids['train']]
    neg_val_pmids = [neg_data.loc[neg_id]['pmid'] for neg_id in neg_ids['val']]
    neg_test_pmids = [neg_data.loc[neg_id]['pmid'] for neg_id in neg_ids['test']]
    del neg_data
    
    pmids_dict[study_design] = {
        'pos': {
            'train': pos_train_pmids,
            'val': pos_val_pmids, 
            'test': neg_test_pmids
        },
        'neg': {
            'train': neg_train_pmids,
            'val': neg_val_pmids,
            'test': neg_test_pmids
        }
    }
    with open(PMIDS, 'wb') as f:
        pkl.dump(pmids_dict, f)
    