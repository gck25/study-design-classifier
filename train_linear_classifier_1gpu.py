import os.path
import gc
import pandas as pd
import pickle as pkl
import numpy as np
import random
from tqdm import tqdm
import json
from utils import get_texts_for_ptype
from sklearn import metrics
from custom_models.custom_model import BatchVectorizerClassifier
from ray import tune, train
import gc


def main():

    model_types = [
        "hinge", 
                   "log_loss", 
                   "modified_huber", 
                   "squared_hinge", 
                   "perceptron"]
    neg_to_pos_ratio = 1

    results_file = "results_titles_abstracts_linear_features1000.json"

    inputs = "titles_abstracts"

    filename_pos_human = "../data/pubmed_articles_pos_human.csv"
    filename_neg_human = "../data/pubmed_articles_neg_human.csv"

    with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)

    ptypes = prop_typ_id_dict.keys()
#     ptypes = ["is_cohort", "is_guideline", "is_consensus", "is_qual", "is_ctprotocol"]

    for ptype in ptypes:
        X_train, X_val, X_test, y_train, y_val, y_test = get_texts_for_ptype(ptype, inputs="titles_abstracts", neg_to_pos_ratio=1, reset_index=True)
        for model_type in model_types:
            
#             config = {
#             "C": tune.uniform(0, 5),
#             "kernel": tune.choice(["linear", "poly", "rbf", "sigmoid", "precomputed"]),
#             "degree": tune.choice([1, 2, 3, 4, 5, 6, 7]),
#             "gamma": tune.choice(["scale", "auto"]),
#             "coef0": tune.uniform(0, 1)
#         }

            config = {
                "penalty": tune.choice(["l2", "l1", "elasticnet"]),
                "alpha": tune.loguniform(1e-10, 1e-2),
            }

            def objective(config, data):
                bvc = BatchVectorizerClassifier(
                    loss=model_type,
                    penalty=config["penalty"],
                    alpha=config["alpha"],
                    l1_ratio=0.15,
                    fit_intercept=True,
                    max_iter=1000,
                    tol=0.001,
                    shuffle=True,
                    verbose=0,
                    epsilon=0.1,
                    n_jobs=None,
                    random_state=None,
                    learning_rate="optimal",
                    eta0=0.0,
                    power_t=0.5,
                    early_stopping=False,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                    class_weight=None,
                    warm_start=False,
                    average=False,
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
            analysis = tune.run(
                tune.with_parameters(objective, data=data),
                config=config,
                num_samples=20,
                resources_per_trial={"gpu": 1},
                verbose=3,
            )

            best_config = analysis.get_best_config(metric="f1", mode="max")
            bvc = BatchVectorizerClassifier(
                loss=model_type,
                penalty=best_config["penalty"],
                alpha=best_config["alpha"],
                l1_ratio=0.15,
                fit_intercept=True,
                max_iter=1000,
                tol=0.001,
                shuffle=True,
                verbose=0,
                epsilon=0.1,
                n_jobs=None,
                random_state=None,
                learning_rate="optimal",
                eta0=0.0,
                power_t=0.5,
                early_stopping=False,
                validation_fraction=0.1,
                n_iter_no_change=5,
                class_weight=None,
                warm_start=False,
                average=False,
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
                "model_type": model_type,
                "neg_to_pos_ratio": neg_to_pos_ratio,
                "best_penalty": best_config["penalty"],
                "best_alpha": best_config["alpha"],
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
