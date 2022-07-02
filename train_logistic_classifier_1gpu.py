import os.path

import pandas as pd
import pickle as pkl
import numpy as np
import random
from tqdm import tqdm
import json
from sklearn import metrics
from custom_models.custom_model import BatchVectorizerClassifier
from ray import tune


def main():
    ptypes = [
        "is_rct",
        "is_sr",
        "is_ct",
    ]
    # ptype = "is_guideline"
    model_types = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
    # model_type = "perceptron"
    neg_to_pos_ratio = 1

    results_file = "classical_ml_results.json"

    filename_pos_human = "../data/pubmed_articles_pos_human.csv"
    filename_neg_human = "../data/pubmed_articles_neg_human.csv"

    with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)

    for ptype in ptypes:
        for model_type in model_types:
            pos_data_human = pd.read_csv(filename_pos_human, chunksize=100000)
            neg_data_human = pd.read_csv(filename_neg_human, chunksize=100000)

            sr_pos_train_ids = prop_typ_id_dict[ptype]["pos"]["train"]
            sr_neg_train_ids = prop_typ_id_dict[ptype]["neg"]["train"]

            sr_pos_val_ids = prop_typ_id_dict[ptype]["pos"]["val"]
            sr_neg_val_ids = prop_typ_id_dict[ptype]["neg"]["val"]

            sr_pos_test_ids = prop_typ_id_dict[ptype]["pos"]["test"]
            sr_neg_test_ids = prop_typ_id_dict[ptype]["neg"]["test"]

            sr_pos_train_set = pd.DataFrame()
            sr_pos_val_set = pd.DataFrame()
            sr_pos_test_set = pd.DataFrame()

            for pos_chunk in tqdm(pos_data_human):

                indices = pos_chunk.index
                indices = list(indices)

                pos_train_indices = list(
                    set(indices).intersection(set(sr_pos_train_ids))
                )
                pos_val_indices = list(set(indices).intersection(set(sr_pos_val_ids)))
                pos_test_indices = list(set(indices).intersection(set(sr_pos_test_ids)))

                sr_pos_train_subset = pos_chunk.loc[pos_train_indices]
                if sr_pos_train_set.empty:
                    sr_pos_train_set = pd.DataFrame(columns=sr_pos_train_set.columns)
                sr_pos_train_set = pd.concat([sr_pos_train_set, sr_pos_train_subset])

                sr_pos_val_subset = pos_chunk.loc[pos_val_indices]
                if sr_pos_val_set.empty:
                    sr_pos_val_set = pd.DataFrame(columns=sr_pos_val_set.columns)
                sr_pos_val_set = pd.concat([sr_pos_val_set, sr_pos_val_subset])

                sr_pos_test_subset = pos_chunk.loc[pos_test_indices]
                if sr_pos_test_set.empty:
                    sr_pos_test_set = pd.DataFrame(columns=sr_pos_test_set.columns)
                sr_pos_test_set = pd.concat([sr_pos_test_set, sr_pos_test_subset])

            sr_pos_train_set[ptype] = int(1)
            sr_pos_val_set[ptype] = int(1)
            sr_pos_test_set[ptype] = int(1)

            del pos_data_human
            del sr_pos_train_subset
            del pos_chunk
            del indices
            del sr_pos_train_ids

            sr_neg_train_ids_sample = random.sample(
                sr_neg_train_ids, round(len(sr_pos_train_set) * neg_to_pos_ratio)
            )

            sr_neg_train_set = pd.DataFrame()
            sr_neg_val_set = pd.DataFrame()
            sr_neg_test_set = pd.DataFrame()

            for neg_chunk in tqdm(neg_data_human):

                indices = neg_chunk.index
                indices = list(indices)
                neg_train_indices = list(
                    set(indices).intersection(set(sr_neg_train_ids_sample))
                )
                neg_val_indices = list(set(indices).intersection(set(sr_neg_val_ids)))
                neg_test_indices = list(set(indices).intersection(set(sr_neg_test_ids)))
                sr_neg_subset = neg_chunk.loc[indices]

                sr_neg_train_subset = neg_chunk.loc[neg_train_indices]
                if sr_neg_train_set.empty:
                    sr_neg_train_set = pd.DataFrame(columns=sr_neg_train_set.columns)
                sr_neg_train_set = pd.concat([sr_neg_train_set, sr_neg_train_subset])

                sr_neg_val_subset = neg_chunk.loc[neg_val_indices]
                if sr_neg_val_set.empty:
                    sr_neg_val_set = pd.DataFrame(columns=sr_neg_val_set.columns)
                sr_neg_val_set = pd.concat([sr_neg_val_set, sr_neg_val_subset])

                sr_neg_test_subset = neg_chunk.loc[neg_test_indices]
                if sr_neg_test_set.empty:
                    sr_neg_test_set = pd.DataFrame(columns=sr_neg_test_set.columns)
                sr_neg_test_set = pd.concat([sr_neg_test_set, sr_neg_test_subset])

            sr_neg_train_set[ptype] = int(0)
            sr_neg_val_set[ptype] = int(0)
            sr_neg_test_set[ptype] = int(0)

            del neg_data_human
            del sr_neg_subset
            del neg_chunk
            del indices

            sr_train_data = pd.concat([sr_neg_train_set, sr_pos_train_set]).sample(
                frac=1
            )
            del sr_neg_train_set
            del sr_pos_train_set
            sr_val_data = pd.concat([sr_neg_val_set, sr_pos_val_set]).sample(frac=1)
            del sr_neg_val_set
            del sr_pos_val_set
            sr_test_data = pd.concat([sr_neg_test_set, sr_pos_test_set]).sample(frac=1)
            del sr_neg_test_set
            del sr_pos_test_set

            y_train = sr_train_data[ptype]
            X_train = sr_train_data["abstract_plaintext"]

            y_val = sr_val_data[ptype]
            X_val = sr_val_data["abstract_plaintext"]

            y_test = sr_test_data[ptype]
            X_test = sr_test_data["abstract_plaintext"]

            del sr_train_data
            del sr_val_data
            del sr_test_data

            config = {
                "loss": tune.grid_search([model_type]),
                "penalty": tune.grid_search(["l2", "l1", "elasticnet"]),
                "alpha": tune.grid_search(list(np.logspace(-10, -5, 20))),
            }

            def objective(config, data):
                bvc = BatchVectorizerClassifier(
                    loss=config["loss"],
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

                tune.report(f1=f1)

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
                resources_per_trial={"gpu": 1},
                verbose=3,
            )

            best_config = analysis.get_best_config(metric="f1", mode="max")
            bvc = BatchVectorizerClassifier(
                loss=best_config["loss"],
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
            recall_val = metrics.recall_score(y_pred_val, y_val)
            precision_val = metrics.precision_score(y_pred_val, y_val)
            f1_val = metrics.f1_score(y_pred_val, y_val)
            accuracy_val = metrics.accuracy_score(y_pred_val, y_val)
            print("Val metrics completed")

            print("Test inference begins")
            y_pred_test = bvc.predict(X_test)
            print("Test inference completed")
            print("Test metrics begins")
            recall_test = metrics.recall_score(y_pred_test, y_test)
            precision_test = metrics.precision_score(y_pred_test, y_test)
            f1_test = metrics.f1_score(y_pred_test, y_test)
            accuracy_test = metrics.accuracy_score(y_pred_test, y_test)
            print("Test metrics ends")

            results_dict = {
                "ptype": ptype,
                "model_type": model_type,
                "neg_to_pos_ratio": neg_to_pos_ratio,
                "best_penalty": best_config["penalty"],
                "best_alpha": best_config["alpha"],
                "recall_val": recall_val,
                "precision_val": precision_val,
                "f1_val": f1_val,
                "accuracy_val": accuracy_val,
                "recall_test": recall_test,
                "precision_test": precision_test,
                "f1_test": f1_test,
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
