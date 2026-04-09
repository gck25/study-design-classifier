import os.path
import gc
import pandas as pd
import pickle as pkl
import numpy as np
import random
from tqdm import tqdm
import json
from sklearn import metrics
from utils import get_texts_for_ptype
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from custom_models.custom_model import BatchVectorizerXGBoost
from ray import tune, train
from ray.train import Checkpoint


def main():

    # model_types = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
    neg_to_pos_ratio = 1

    results_file = "results_titles_abstracts_gbdt_features1000.json"

    with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)

#     ptypes = prop_typ_id_dict.keys()
    ptypes = [
#         "is_rct", 
        "is_sr", "is_casecontrol", "is_cohort", "is_guideline"]

    for ptype in ptypes:
        X_train, X_val, X_test, y_train, y_val, y_test = get_texts_for_ptype(ptype, inputs="titles_abstracts", neg_to_pos_ratio=1, reset_index=True)

        config = {
            "objective": tune.choice(
                [
                    "binary:hinge",
                    "binary:logistic",
                    "binary:logitraw",
                ]
            ),
            "max_depth": tune.randint(1, 18),
            "subsample": tune.uniform(0.5, 1.0),
            "gamma": tune.uniform(0, 10),
            "reg_alpha": tune.loguniform(1e-5, 1e5),
            "reg_lambda": tune.loguniform(1e-5, 1e5),
            "colsample_bytree": tune.uniform(0.5, 1),
            "min_child_weight": tune.uniform(0, 10),
            "eta": tune.loguniform(1e-4, 1e-1)
        }

        

        def objective(config, data):
            bvc = BatchVectorizerXGBoost(
                # objective=config["objective"],
                objective="binary:logistic",
                # max_depth=config["max_depth"],
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
            bvc.fit(data["X_train"], data["y_train"])
            y_pred_val = bvc.predict(data["X_val"])
            f1 = metrics.f1_score(y_pred_val, data["y_val"])
            metric_dict = {
                "f1": f1
            }
            train.report(metric_dict)

            return {"f1": f1}

        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }
#         algo = BayesOptSearch(space, metric="f1", mode="max")
        # algo = ConcurrencyLimiter(algo, max_concurrent=60)
        analysis = tune.run(
            tune.with_parameters(objective, data=data),
            config=config,
            num_samples=20,
#             search_alg=algo,
            resources_per_trial={"gpu": 1},
            verbose=3,
        )

        best_config = analysis.get_best_config(metric="f1", mode="max")
        bvc = BatchVectorizerXGBoost(
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
