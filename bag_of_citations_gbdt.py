import os
import json
import copy
import requests
import numpy as np
import pickle as pkl
import xgboost as xgb
from tqdm import tqdm
from random import sample, choice, shuffle
import scipy
from scipy import sparse
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import metrics
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
import importlib
import open_alex_utils
from open_alex_utils import get_pos_citation_dataset, get_neg_citation_dataset, extract_bow_feats, get_bow_inputs_labels
from custom_models.custom_model import BatchVectorizerClassifierCitationsNonTexts

from ray import tune
from ray import train
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import ASHAScheduler


def main():

    file_list = [file for file in os.listdir('open_alex_citations') if 'open' in file]
    ids_file = '../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl'
    results_file = "results_bag_of_citations_gbdt.json"
    
    neg_to_pos_ratio = 1
    
    PMIDS = '../data/pmids.pkl'
    with open(PMIDS, 'rb') as f:
        pmids = pkl.load(f)
    
    with open(ids_file, "rb") as f:
        prop_typ_id_dict = pkl.load(f)

    ptypes = prop_typ_id_dict.keys()
    
    def objective(config, data):
        bvc = BatchVectorizerClassifierCitationsNonTexts(
            objective="binary:logistic",
            max_depth=18,
            gamma=config["gamma"],
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
            colsample_bytree=config["colsample_bytree"],
            min_child_weight=config["min_child_weight"],
            n_estimators=180,
            seed=0,
        )
        bvc.fit(data["X_train"], data["y_train"])
        y_pred_val = bvc.predict(data["X_val"])
        f1 = metrics.f1_score(y_pred_val, data["y_val"])
        
        train.report({"f1": f1})
        config = {
            "gamma": tune.uniform(0, 10),
            "reg_alpha": tune.loguniform(10**-5, 10**5),
            "reg_lambda": tune.loguniform(10**-5, 10**5),
            "colsample_bytree": tune.uniform(0.5, 1),
            "min_child_weight": tune.uniform(0, 10),
        }
    
    for ptype in tqdm(ptypes):
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
        
        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }
        
        config = {
            "gamma": tune.uniform(0, 10),
            "reg_alpha": tune.loguniform(10**-5, 10**5),
            "reg_lambda": tune.loguniform(10**-5, 10**5),
            "colsample_bytree": tune.uniform(0.5, 1),
            "min_child_weight": tune.uniform(0, 10),
        }
        
        algo = BayesOptSearch(metric="f1", mode="max")
        # algo = ConcurrencyLimiter(algo, max_concurrent=60)
        analysis = tune.run(
            tune.with_parameters(objective, data=data),
            config=config,
            num_samples=20,
            search_alg=algo,
            resources_per_trial={"gpu": 1},
            verbose=3,
        )
        
        best_config = analysis.get_best_config(metric="f1", mode="max")
        
        bvc = BatchVectorizerClassifierCitationsNonTexts(
            objective="binary:logistic",
            max_depth=18,
            gamma=float(best_config["gamma"]),
            reg_alpha=float(best_config["reg_alpha"]),
            reg_lambda=float(best_config["reg_lambda"]),
            colsample_bytree=float(best_config["colsample_bytree"]),
            min_child_weight=float(best_config["min_child_weight"]),
            n_estimators=180,
            seed=0,
        )
        
        print("Fitting begins")
        bvc.fit(X_train, y_train)
        print("Fitting completed")

        print("Val inference begins")
        y_pred_val = bvc.predict(X_val)
        print("Val inference completed")

        print("Val metrics begins")
        recall_val = metrics.recall_score(y_val, y_pred_val, pos_label=1)
        spec_val = metrics.recall_score(y_val, y_pred_val, pos_label=0)
        precision_val = metrics.precision_score(y_val, y_pred_val)
        f1_val = metrics.f1_score(y_val, y_pred_val)
        accuracy_val = metrics.accuracy_score(y_val, y_pred_val)
        print("Val metrics completed")

        print("Test inference begins")
        y_pred_test = bvc.predict(X_test)
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
            
            
if __name__ == "__main__":
    main()