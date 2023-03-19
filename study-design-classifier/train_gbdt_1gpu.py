import os.path

import pandas as pd
import pickle as pkl
import numpy as np
import random
from tqdm import tqdm
import json
from sklearn import metrics
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch
from custom_models.custom_model import BatchVectorizerXGBoost
from ray import tune


def main():

    # model_types = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
    neg_to_pos_ratio = 1

    results_file = "results_titles_abstracts_gbdt.json"

    inputs = "titles_abstracts"

    filename_pos_human = "../data/pubmed_articles_pos_human.csv"
    filename_neg_human = "../data/pubmed_articles_neg_human.csv"

    with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)

    ptypes = prop_typ_id_dict.keys()
    # ptypes = ["is_rct", "is_sr", "is_casecontrol", "is_ct"]

    for ptype in ptypes:
        # for model_type in model_types:
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

            pos_train_indices = list(set(indices).intersection(set(sr_pos_train_ids)))
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

        sr_train_data = pd.concat([sr_neg_train_set, sr_pos_train_set]).sample(frac=1)
        del sr_neg_train_set
        del sr_pos_train_set
        sr_val_data = pd.concat([sr_neg_val_set, sr_pos_val_set]).sample(frac=1)
        del sr_neg_val_set
        del sr_pos_val_set
        sr_test_data = pd.concat([sr_neg_test_set, sr_pos_test_set]).sample(frac=1)
        del sr_neg_test_set
        del sr_pos_test_set

        y_train = sr_train_data[ptype]
        y_val = sr_val_data[ptype]
        y_test = sr_test_data[ptype]
        if inputs == "abstracts":
            X_train = sr_train_data["abstract_plaintext"]
            X_val = sr_val_data["abstract_plaintext"]
            X_test = sr_test_data["abstract_plaintext"]
        elif inputs == "titles":
            X_train = sr_train_data["title"]
            X_val = sr_val_data["title"]
            X_test = sr_test_data["title"]
        elif inputs == "titles_abstracts":
            X_train = pd.Series(
                [
                    "{0} {1}".format(train_title, train_abstract)
                    for train_title, train_abstract in zip(
                        sr_train_data["title"], sr_train_data["abstract_plaintext"]
                    )
                ]
            )
            X_val = pd.Series(
                [
                    "{0} {1}".format(val_title, val_abstract)
                    for val_title, val_abstract in zip(
                        sr_val_data["title"], sr_val_data["abstract_plaintext"]
                    )
                ]
            )
            X_test = pd.Series(
                [
                    "{0} {1}".format(test_title, test_abstract)
                    for test_title, test_abstract in zip(
                        sr_test_data["title"], sr_test_data["abstract_plaintext"]
                    )
                ]
            )

        del sr_train_data
        del sr_val_data
        del sr_test_data

        # config = {
        #     "objective": tune.grid_search(
        #         [
        #             "binary:hinge",
        #             "binary:logistic",
        #             "binary:logitraw",
        #         ]
        #     ),
        #     "max_depth": tune.grid_search(list(np.linspace(3, 18, 16).astype(int))),
        #     "gamma": tune.grid_search(list(np.linspace(0, 10, 10))),
        #     "reg_alpha": tune.grid_search(list(np.logspace(-5, 5, 10))),
        #     "reg_lambda": tune.grid_search(list(np.logspace(-5, 5, 10))),
        #     "colsample_bytree": tune.grid_search(list(np.linspace(0.5, 1, 5))),
        #     "min_child_weight": tune.grid_search(list(np.linspace(0, 10, 10))),
        #     "eta": tune.grid_search(list(np.logspace(-5, 0, 10))),
        #     "n_estimators": tune.grid_search([50, 65, 80, 100, 115, 130, 150, 180]),
        #     "seed": tune.grid_search([0]),
        # }

        space = {
            "gamma": (0, 10),
            "reg_alpha": (0, 1000),
            "reg_lambda": (0, 1000),
            "colsample_bytree": (0.5, 1),
            "min_child_weight": (0, 10),
            "eta": (-5, 0),
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
                # n_estimators=config["n_estimators"],
                n_estimators=180,
                seed=0,
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
        algo = BayesOptSearch(space, metric="f1", mode="max")
        # algo = ConcurrencyLimiter(algo, max_concurrent=60)
        analysis = tune.run(
            tune.with_parameters(objective, data=data),
            # config=config,
            num_samples=20,
            search_alg=algo,
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
        recall_val = metrics.recall_score(y_val, y_pred_val)
        precision_val = metrics.precision_score(y_val, y_pred_val)
        f1_val = metrics.f1_score(y_val, y_pred_val)
        accuracy_val = metrics.accuracy_score(y_val, y_pred_val)
        print("Val metrics completed")

        print("Test inference begins")
        y_pred_test = bvc.predict(X_test)
        print("Test inference completed")
        print("Test metrics begins")
        recall_test = metrics.recall_score(y_test, y_pred_test)
        precision_test = metrics.precision_score(y_test, y_pred_test)
        f1_test = metrics.f1_score(y_test, y_pred_test)
        accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
        print("Test metrics ends")

        results_dict = {
            "ptype": ptype,
            "model_type": "gbdt",
            "neg_to_pos_ratio": neg_to_pos_ratio,
            # "best_objective": best_config["objective"],
            # "max_depth": best_config["max_depth"],
            "gamma": best_config["gamma"],
            "reg_alpha": best_config["reg_alpha"],
            "reg_lambda": best_config["reg_lambda"],
            "colsample_bytree": best_config["colsample_bytree"],
            "min_child_weight": best_config["min_child_weight"],
            # "n_estimators": best_config["n_estimators"],
            # "seed": best_config["seed"],
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
