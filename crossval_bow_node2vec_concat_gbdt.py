import os
import gc
import json
import random
import pickle as pkl
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from fastnode2vec import Node2Vec, Graph
from utils import get_texts_for_ptype_pmid, get_embeddings, get_texts_citations_labels

from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray import tune, train
from ray.train import Checkpoint
from custom_models.custom_model import BatchVectorizerXGBoostConcatCrossVal


def main():
    results_file = "results_bow_node2vec_concat_gbdt.json"
    metrics_file = "results_crossvals_bow_node2vec_concat_gbdt.json"
    
    with open(results_file, "r") as f:
        results_list = json.load(f)
    
    inputs = "titles_abstracts"
    
    neg_to_pos_ratio = 1
    results_file = "results_bow_node2vec_concat_gbdt.json"
    MODEL_DIR = "../models"
    EMBEDDING_FILENAME = os.path.join(MODEL_DIR, 'node_model.pkl')
    PMIDS = '../data/pmids.pkl'
    
    with open("../data/graph_lst.pkl", "rb") as f:
        graph_lst = pkl.load(f)
    
    with open("../data/pmid_to_openAlex.json") as f:
        pmid_to_openAlex = json.load(f)
    
    with open(PMIDS, 'rb') as f:
        pmids = pkl.load(f)
        
    graph = Graph(graph_lst, directed=False, weighted=False)
    node2vec = Node2Vec(graph, dim=10, walk_length=100, context=10, p=2.0, q=0.5, workers=5)  # Use temp_folder for big graphs
    model = node2vec.wv.load_word2vec_format(EMBEDDING_FILENAME)
    
    metrics_list = []
    for results_json in tqdm(results_list):
        ptype = results_json["ptype"]
        
        pos_train_embeddings, neg_train_embeddings = get_embeddings(pmids, pmid_to_openAlex, ptype, model, split='train')
        pos_val_embeddings, neg_val_embeddings = get_embeddings(pmids, pmid_to_openAlex, ptype, model, split='val')
        pos_test_embeddings, neg_test_embeddings = get_embeddings(pmids, pmid_to_openAlex, ptype, model, split='test')
        
        pos_train_pmids = list(pos_train_embeddings.keys())
        neg_train_pmids = list(neg_train_embeddings.keys())
        pos_val_pmids = list(pos_val_embeddings.keys())
        neg_val_pmids = list(neg_val_embeddings.keys())
        pos_test_pmids = list(pos_test_embeddings.keys())
        neg_test_pmids = list(neg_test_embeddings.keys())
        pmids_to_use = {
            "pos_train": pos_train_pmids,
            "neg_train": neg_train_pmids,
            "pos_val": pos_val_pmids,
            "neg_val": neg_val_pmids,
            "pos_test": pos_test_pmids,
            "neg_test": neg_test_pmids
        }
        
        train_set, val_set, test_set = get_texts_for_ptype_pmid(ptype, inputs="titles_abstracts", neg_to_pos_ratio=1, reset_index=True)
        
        train_set_concat = {}
        for pmid in pos_train_pmids:
            train_set_concat[pmid] = {}
            train_set_concat[pmid]["X_graph"] = pos_train_embeddings[pmid]
            train_set_concat[pmid]["X_text"] = train_set[pmid]["X"]
            train_set_concat[pmid]["y"] = 1

        for pmid in neg_train_pmids:
            train_set_concat[pmid] = {}
            train_set_concat[pmid]["X_graph"] = neg_train_embeddings[pmid]
            train_set_concat[pmid]["X_text"] = train_set[pmid]["X"]
            train_set_concat[pmid]["y"] = 0
            
        train_df = get_texts_citations_labels(train_set_concat)
        train_df = train_df.sample(frac=1)
        
        X = train_df
        y = train_df["labels"]
        
        bvc = BatchVectorizerXGBoostConcatCrossVal(
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
        
#         random_state = 42
#         rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=random_state)
        
        scores = cross_validate(bvc, X, y, cv=10,
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