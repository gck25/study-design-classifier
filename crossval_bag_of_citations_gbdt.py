import gc
import os.path
import pandas as pd
import pickle as pkl
import numpy as np
import random

from collections import defaultdict
from tqdm import tqdm
import json
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from functools import partial
import open_alex_utils
from open_alex_utils import get_pos_citation_dataset, get_neg_citation_dataset, extract_bow_feats, get_bow_inputs_labels
from custom_models.custom_model import BatchVectorizerClassifierCitationsNonTexts
from utils import get_texts_for_ptype


def main():
    neg_to_pos_ratio = 1
    
    file_list = [file for file in os.listdir('open_alex_citations') if 'open' in file]
    ids_file = '../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl'

    results_file = "results_bag_of_citations_gbdt.json"
    metrics_file = "results_crossval_bag_of_citations_gbdt.json"
    
    with open(results_file, "r") as f:
        results_list = json.load(f)
    
    PMIDS = '../data/pmids.pkl'
    with open(PMIDS, 'rb') as f:
        pmids = pkl.load(f)
    
    with open(ids_file, "rb") as f:
        prop_typ_id_dict = pkl.load(f)

    ptypes = prop_typ_id_dict.keys()
    
    
    metrics_list = []
    for results_json in tqdm(results_list):
        
        ptype = results_json["ptype"]
        
        pos_train_set, pos_train_feat_lst = get_pos_citation_dataset(pmids[ptype]['pos']['train'])
        pos_val_set, pos_val_feat_lst = get_pos_citation_dataset(pmids[ptype]['pos']['val'])
        pos_test_set, pos_test_feat_lst = get_pos_citation_dataset(pmids[ptype]['pos']['test'])

        neg_train_set, neg_train_feat_lst = get_neg_citation_dataset(pmids[ptype]['neg']['train'], len(pos_train_set))
        neg_val_set, neg_val_feat_lst = get_neg_citation_dataset(pmids[ptype]['neg']['val'])
        neg_test_set, neg_test_feat_lst = get_neg_citation_dataset(pmids[ptype]['neg']['test'])
        
        feat_lst = pos_train_feat_lst + pos_val_feat_lst + pos_test_feat_lst + neg_train_feat_lst + neg_val_feat_lst + neg_test_feat_lst 
        
        
        freqs = defaultdict(int)

        for paper in tqdm(pos_train_set):
            for opid in paper:
                freqs[opid] +=1

        for paper in tqdm(neg_train_set):
            for opid in paper:
                freqs[opid] +=1
                
        freqs = {k: v for k, v in sorted(freqs.items(), key=lambda item: item[1], reverse=True)}
        
        freqs_reduced = {k: v for k, v in freqs.items() if  v > 3}
        
        feature_lst_reduced = list(freqs_reduced)
        
        bow_neg_train = extract_bow_feats(neg_train_set, feature_lst_reduced)
        bow_neg_val = extract_bow_feats(neg_val_set, feature_lst_reduced)
        bow_neg_test = extract_bow_feats(neg_test_set, feature_lst_reduced)
        bow_pos_train = extract_bow_feats(pos_train_set, feature_lst_reduced)
        bow_pos_val = extract_bow_feats(pos_val_set, feature_lst_reduced)
        bow_pos_test = extract_bow_feats(pos_test_set, feature_lst_reduced)

        
        X_train, y_train = get_bow_inputs_labels(bow_pos_train, bow_neg_train, is_shuffle = True)
        X_val, y_val = get_bow_inputs_labels(bow_pos_val, bow_neg_val, is_shuffle = True)
        X_test, y_test = get_bow_inputs_labels(bow_pos_test, bow_neg_test, is_shuffle = True)
        
        bvc = BatchVectorizerClassifierCitationsNonTexts(
            objective="binary:logistic",
            # max_depth=config["max_depth"],
            max_depth=18,
            gamma=results_json["gamma"],
            reg_alpha=results_json["reg_alpha"],
            reg_lambda=results_json["reg_lambda"],
            colsample_bytree=results_json["colsample_bytree"],
            min_child_weight=results_json["min_child_weight"],
            # n_estimators=config["n_estimators"],
            n_estimators=180,
            seed=0,
        )
        def recall(y_true, y_pred, labels=None, average='binary', sample_weight=None, zero_division='warn'):
            return metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        
        def specificity(y_true, y_pred, labels=None, average='binary', sample_weight=None, zero_division='warn'):
            return metrics.recall_score(y_true, y_pred, labels=None, pos_label=0, average='binary', sample_weight=None, zero_division='warn')
        
        scores = cross_validate(bvc, X_train, y_train, cv=10,
                                scoring={
                                    "recall": make_scorer(recall), 
                                    "spec": make_scorer(specificity), 
                                    "prec": make_scorer(metrics.precision_score), 
                                    "f1": make_scorer(metrics.f1_score), 
                                    "acc": make_scorer(metrics.accuracy_score),
                                },
                                return_train_score=True)
        
        metrics_list.append({
            "ptype": ptype,
            "model_type": results_json["model_type"],
            "scores": {k:list(v) for k,v in scores.items()}
        })
        
        with open(metrics_file, "w") as f:
            json.dump(metrics_list, f)
        
        del bvc
        gc.collect()
    
    
if __name__ == "__main__":
    main()
        
   