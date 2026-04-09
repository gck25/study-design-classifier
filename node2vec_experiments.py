import os
import json
import xgboost as xgb
import pickle as pkl
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from fastnode2vec import Node2Vec, Graph
from gensim.models import Word2Vec
from sklearn import metrics
from datasets import load_metric
from ray import tune
from ray import train
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import ASHAScheduler


def get_embeddings(pmids, pmid_to_openAlex, study_design, model, split='train'):
    
    pos_pmids = pmids[study_design]['pos'][split]
    pos_embeddings = []

    for pmid in tqdm(pos_pmids):
        try:
            openAlexID = pmid_to_openAlex[str(pmid)]
            embedding = model.get_vector(openAlexID)
            pos_embeddings.append(embedding)
        except:
            continue

    neg_pmids = pmids[study_design]['neg'][split]
    neg_embeddings = []
    for pmid in tqdm(neg_pmids):
        try:
            openAlexID = pmid_to_openAlex[str(pmid)]
            embedding = model.get_vector(openAlexID)
            neg_embeddings.append(embedding)
        except:
            continue
    return pos_embeddings, neg_embeddings

def main():
    
    results_file = "results_node2vec_gbdt_sample.json"
    MODEL_DIR = "../models"
    EMBEDDING_FILENAME = os.path.join(MODEL_DIR, 'node_model.pkl')
    PMIDS = '../data/pmids.pkl'  
    neg_to_pos_ratio = 1
    
    metric_acc = load_metric("accuracy")
    metric_p = load_metric("precision")
    metric_r = load_metric("recall")
    metric_f = load_metric("f1")
    

    with open('../data/open_alex_graph.json') as f:
        oalex_graph_dict = json.load(f)
    
    with open("../data/pmid_to_openAlex.json") as f:
        pmid_to_openAlex = json.load(f)
    
    with open("../data/pmid2node_id_sample.json", "r") as f:
        pmid2node_id = json.load(f)
        
    openAlex_included = [v for k,v in tqdm(pmid_to_openAlex.items()) if k in pmid2node_id and v in oalex_graph_dict]

    graph_lst = []
    for k in tqdm(openAlex_included):
        values = list(set(oalex_graph_dict[k])& set(openAlex_included))
        for v in values:
            graph_lst.append((k, v))

    graph = Graph(graph_lst, directed=False, weighted=False)

    node2vec = Node2Vec(graph, dim=10, walk_length=100, context=10, p=2.0, q=0.5, workers=5)  # Use temp_folder for big graphs


    # Load model after Node2Vec.wv.save_word2vec_format
    model = node2vec.wv.load_word2vec_format(EMBEDDING_FILENAME)
  
    with open(PMIDS, 'rb') as f:
        pmids = pkl.load(f)

    

    for study_design in tqdm(pmids.keys()):
        print(study_design)
        
        pos_train_embeddings, neg_train_embeddings = get_embeddings(pmids, pmid_to_openAlex, study_design, model, split='train')
        
        print("Positive train length", len(pos_train_embeddings))
        print("Negative train length", len(neg_train_embeddings))
        
        pos_df = pd.DataFrame({"x": pos_train_embeddings})
        pos_df['y'] = 1

        neg_df = pd.DataFrame({"x": neg_train_embeddings})
        neg_df['y'] = 0

        train_df = pd.concat([pos_df, neg_df]).sample(frac=1)
        X_train = np.array([list(el) for el in tqdm(list(train_df['x']))])
        y_train = train_df['y']
        
        pos_val_embeddings, neg_val_embeddings = get_embeddings(pmids, pmid_to_openAlex, study_design, model, split='val')
        
        print("Positive val length", len(pos_val_embeddings))
        print("Negative val length", len(neg_val_embeddings))
        
        pos_df = pd.DataFrame({"x": pos_val_embeddings})
        pos_df['y'] = 1

        neg_df = pd.DataFrame({"x": neg_val_embeddings})
        neg_df['y'] = 0

        val_df = pd.concat([pos_df, neg_df]).sample(frac=1)
        X_val = np.array([list(el) for el in tqdm(list(val_df['x']))])
        y_val = val_df['y']

        data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }

        def objective(config, data):
            bvc = xgb.XGBClassifier(
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
        bvc = xgb.XGBClassifier(
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

        pos_test_embeddings, neg_test_embeddings = get_embeddings(pmids, pmid_to_openAlex, study_design, model, split='test')
        
        print("Positive test length", len(pos_test_embeddings))
        print("Negative test length", len(neg_test_embeddings))
        
        pos_df = pd.DataFrame({"x": pos_test_embeddings})
        pos_df['y'] = 1

        neg_df = pd.DataFrame({"x": neg_test_embeddings})
        neg_df['y'] = 0

        test_df = pd.concat([pos_df, neg_df]).sample(frac=1)
        X_test = np.array([list(el) for el in tqdm(list(test_df['x']))])
        y_test = test_df['y']
    
    
        print("Fitting begins")
        bvc.fit(X_train, train_df['y'])
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
                "ptype": study_design,
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
    
    