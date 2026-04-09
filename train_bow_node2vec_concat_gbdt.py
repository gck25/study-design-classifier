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
from fastnode2vec import Node2Vec, Graph
from utils import get_texts_for_ptype_pmid, get_embeddings, get_texts_citations_labels

from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray import tune, train
from ray.train import Checkpoint
from custom_models.custom_model import BatchVectorizerXGBoostConcat

def main():
    neg_to_pos_ratio = 1
    results_file = "results_bow_node2vec_concat_gbdt.json"
    MODEL_DIR = "../models"
    EMBEDDING_FILENAME = os.path.join(MODEL_DIR, 'node_model.pkl')
    PMIDS = '../data/pmids.pkl'

    with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)
    ptypes = prop_typ_id_dict.keys()
    
    
    
    with open("../data/graph_lst.pkl", "rb") as f:
        graph_lst = pkl.load(f)
    
    with open("../data/pmid_to_openAlex.json") as f:
        pmid_to_openAlex = json.load(f)
    
    with open(PMIDS, 'rb') as f:
        pmids = pkl.load(f)
    
    
    
    graph = Graph(graph_lst, directed=False, weighted=False)
    node2vec = Node2Vec(graph, dim=10, walk_length=100, context=10, p=2.0, q=0.5, workers=5)  # Use temp_folder for big graphs
    model = node2vec.wv.load_word2vec_format(EMBEDDING_FILENAME)
    
    config = {
        "subsample": tune.uniform(0.5, 1.0),
        "gamma": tune.uniform(0, 10),
        "reg_alpha": tune.loguniform(1e-5, 1e5),
        "reg_lambda": tune.loguniform(1e-5, 1e5),
        "colsample_bytree": tune.uniform(0.5, 1),
        "min_child_weight": tune.uniform(0, 10),
        "eta": tune.loguniform(1e-4, 1e-1)
    }
    
    def objective(config, data):
        bvc = BatchVectorizerXGBoostConcat(
            objective="binary:logistic",
            max_depth=18,
            gamma=config["gamma"],
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
            colsample_bytree=config["colsample_bytree"],
            min_child_weight=config["min_child_weight"],
            subsample=config["subsample"],
            n_estimators=180,
            seed=0,
        )
        bvc.fit(data["train_texts"], data["train_node_embeds"], data["train_labels"])
        y_pred_val = bvc.predict(data["val_texts"], data["val_node_embeds"])
        f1 = metrics.f1_score(y_pred_val, data["val_labels"])
        metric_dict = {
            "f1": f1
        }
        train.report(metric_dict)
        return {"f1": f1}
    
    for ptype in ptypes:
        
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

        val_set_concat = {}
        for pmid in pos_val_pmids:
            val_set_concat[pmid] = {}
            val_set_concat[pmid]["X_graph"] = pos_val_embeddings[pmid]
            val_set_concat[pmid]["X_text"] = val_set[pmid]["X"]
            val_set_concat[pmid]["y"] = 1

        for pmid in neg_val_pmids:
            val_set_concat[pmid] = {}
            val_set_concat[pmid]["X_graph"] = neg_val_embeddings[pmid]
            val_set_concat[pmid]["X_text"] = val_set[pmid]["X"]
            val_set_concat[pmid]["y"] = 0

        test_set_concat = {}
        for pmid in pos_test_pmids:
            test_set_concat[pmid] = {} 
            test_set_concat[pmid]["X_graph"] = pos_test_embeddings[pmid]
            test_set_concat[pmid]["X_text"] = test_set[pmid]["X"]
            test_set_concat[pmid]["y"] = 1

        for pmid in neg_test_pmids:
            test_set_concat[pmid] = {}
            test_set_concat[pmid]["X_graph"] = neg_test_embeddings[pmid]
            test_set_concat[pmid]["X_text"] = test_set[pmid]["X"]
            test_set_concat[pmid]["y"] = 0
        
       
        
        train_df = get_texts_citations_labels(train_set_concat)
        train_df = train_df.sample(frac=1)
        
        val_df = get_texts_citations_labels(val_set_concat)
        val_df = val_df.sample(frac=1)
        
        test_df = get_texts_citations_labels(test_set_concat)
        test_df = test_df.sample(frac=1)
        
        data = {
            "train_texts": train_df["texts"],
            "train_node_embeds": train_df["node_embeds"],
            "train_labels": train_df["labels"],
            "val_texts": val_df["texts"],
            "val_node_embeds": val_df["node_embeds"],
            "val_labels": val_df["labels"]
        }
        
        analysis = tune.run(
            tune.with_parameters(objective, data=data),
            config=config,
            num_samples=20,
            resources_per_trial={"gpu": 1},
            verbose=3,
        )
        
        best_config = analysis.get_best_config(metric="f1", mode="max")
        bvc = BatchVectorizerXGBoostConcat(
            objective="binary:logistic",
            # max_depth=config["max_depth"],
            max_depth=18,
            gamma=best_config["gamma"],
            reg_alpha=best_config["reg_alpha"],
            reg_lambda=best_config["reg_lambda"],
            colsample_bytree=best_config["colsample_bytree"],
            min_child_weight=best_config["min_child_weight"],
            subsample=config["subsample"],
            # n_estimators=config["n_estimators"],
            n_estimators=180,
            seed=0,
        )

        del analysis

        print("Fitting begins")
        bvc.fit(train_df["texts"], train_df["node_embeds"], train_df["labels"])
        print("Fitting completed")

        print("Val inference begins")
        y_pred_val = bvc.predict(val_df["texts"], val_df["node_embeds"])
        y_val = val_df["labels"]
        print("Val inference completed")

        print("Val metrics begins")
        recall_val = metrics.recall_score(y_val, y_pred_val, pos_label=1)
        spec_val = metrics.recall_score(y_val, y_pred_val, pos_label=0)
        precision_val = metrics.precision_score(y_val, y_pred_val)
        f1_val = metrics.f1_score(y_val, y_pred_val)
        accuracy_val = metrics.accuracy_score(y_val, y_pred_val)
        print("Val metrics completed")

        print("Test inference begins")
        
        y_pred_test = bvc.predict(test_df["texts"], test_df["node_embeds"])
        y_test = test_df["labels"]
        print("Test inference completed")
        print("Test metrics begins")
        recall_test = metrics.recall_score(y_test, y_pred_test, pos_label=1)
        spec_test = metrics.recall_score(y_test, y_pred_test, pos_label=0)
        precision_test = metrics.precision_score(y_test, y_pred_test)
        f1_test = metrics.f1_score(y_test, y_pred_test)
        accuracy_test = metrics.accuracy_score(y_test, y_pred_test)   
        print("Test metrics ends")

        results_dict = { 
                "ptype": ptype,
                "model_type": "gbdt",
                "neg_to_pos_ratio": neg_to_pos_ratio,
                "gamma": best_config["gamma"],
                "reg_alpha": best_config["reg_alpha"],
                "reg_lambda": best_config["reg_lambda"],
                "colsample_bytree": best_config["colsample_bytree"],
                "min_child_weight": best_config["min_child_weight"],
                "subsample": best_config["subsample"],
                "recall_val": recall_val,
                "spec_val": spec_val,
                "precision_val_even": precision_val,
                "f1_val_even": f1_val,
                "accuracy_val": accuracy_val,
                "recall_test": recall_test,
                "spec_test": spec_test,
                "precision_test_even": precision_test,
                "f1_test_even": f1_test,
                "accuracy_test": accuracy_test,
        }
        
        
        if os.path.exists(results_file):
            with open(results_file, "r") as f_read:
                results_list = json.load(f_read)
        else:
            results_list = []

        results_list.append(results_dict)
        with open(results_file, "w") as f_write:
            json.dump(results_list, f_write)
        
        del bvc
        gc.collect()

if __name__ == "__main__":
    main()
        
        
        