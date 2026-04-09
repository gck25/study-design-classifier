import gc
import os.path
import pandas as pd
import pickle as pkl
import numpy as np
import random
from tqdm import tqdm
import json
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from functools import partial
from custom_models.custom_model import BatchVectorizerXGBoost
from utils import get_texts_for_ptype

                

def main():
    neg_to_pos_ratio = 1

    results_file = "results_titles_abstracts_gbdt_features1000.json"
    metrics_file = "results_crossval_gbdt_features1000.json"
    
    with open(results_file, "r") as f:
        results_list = json.load(f)
    
    
    
    
    metrics_list = []
    for results_json in tqdm(results_list):
        ptype = results_json["ptype"]
        X_train, X_val, X_test, y_train, y_val, y_test = get_texts_for_ptype(ptype, inputs="titles_abstracts", neg_to_pos_ratio=neg_to_pos_ratio)
        
        
        bvc = BatchVectorizerXGBoost(
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