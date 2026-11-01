import gc
import os.path
import pandas as pd
import pickle as pkl
import numpy as np
import random
from tqdm import tqdm
import json
import xgboost as xgb
from sklearn import metrics
from scipy.optimize import minimize_scalar
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from functools import partial
from fastnode2vec import Node2Vec, Graph
from gensim.models import Word2Vec
from custom_models.custom_model import BatchVectorizerXGBoost
from utils import get_texts_for_ptype_pmid, get_embeddings, get_citations_labels


def main():
    neg_to_pos_ratio = 1
    results_file = "results_node2vec_gbdt_sample.json"
    node2vec_gbdt_metrics_adjusted_results_path = "results_adjusted_metrics_node2vec_gbdt.json"
    results_node2vec_gbdt_file = "results_node2vec_gbdt_sample.json"
    results_texts_gbdt_file = "results_titles_abstracts_gbdt_features1000.json"
    MODEL_DIR = "../models"
    EMBEDDING_FILENAME = os.path.join(MODEL_DIR, 'node_model.pkl')
    PMIDS = '../data/pmids.pkl'
    
    with open(results_file, "r") as f:
        results_list = json.load(f)
        
    with open(results_node2vec_gbdt_file, "r") as f:
        results_node2vec_gbdt_list = json.load(f)
        
    with open(PMIDS, 'rb') as f:
        pmids = pkl.load(f)
    
    with open("../data/pmid_to_openAlex.json") as f:
        pmid_to_openAlex = json.load(f)
    
    with open("../data/graph_lst.pkl", "rb") as f:
        graph_lst = pkl.load(f)

    graph = Graph(graph_lst, directed=False, weighted=False)
    node2vec = Node2Vec(graph, dim=10, walk_length=100, context=10, p=2.0, q=0.5, workers=5)  # Use temp_folder for big graphs

    # Load model after Node2Vec.wv.save_word2vec_format
    model = node2vec.wv.load_word2vec_format(EMBEDDING_FILENAME)
    def predict(df, db=0):
        return [int(x>=db) for x in df ]
    
    def get_scores(predictions):
        return [np.exp(p[1])/(np.exp(p[0]) + np.exp(p[1])) for p in predictions]
    
    neg_pos_ratio_dict = {
        "is_rct": 5585042/489295,
        "is_sr": 5659292/121063,
        "is_casecontrol": 5628219/65790,
        "is_cohort": 5632460/78688,
        "is_guideline": 5676411/33826
    }

    def calc_precision(recall, specificity, neg_pos_ratio):
        return recall / (recall + neg_pos_ratio * (1 - specificity) + 1e-15)

    def calc_f1(recall, precision):
        return 2*recall*precision/(recall+precision + 1e-15)
    
    metrics_lst = []

    for results_node2vec in tqdm(results_node2vec_gbdt_list):
        ptype = results_node2vec['ptype']
        
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
            
        train_df = get_citations_labels(train_set_concat)
        train_df = train_df.sample(frac=1)
            
        val_df = get_citations_labels(val_set_concat)
        val_df = val_df.sample(frac=1)
            
        test_df = get_citations_labels(test_set_concat)
        test_df = test_df.sample(frac=1)
        
        bvc_node2vec = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=18,
            gamma=results_node2vec["gamma"],
            reg_alpha=results_node2vec["reg_alpha"],
            reg_lambda=results_node2vec["reg_lambda"],
            colsample_bytree=results_node2vec["colsample_bytree"],
            min_child_weight=results_node2vec["min_child_weight"],
            n_estimators=180,
            seed=0,
        )
        
        bvc_node2vec.fit(np.vstack(train_df["node_embeds"]), np.array(train_df["labels"]))
        
        val_node2vec_probs = bvc_node2vec.predict_proba(np.vstack(val_df["node_embeds"]))
        val_node2vec_probs = np.array([pos for neg, pos in val_node2vec_probs])
        
        test_node2vec_probs = bvc_node2vec.predict_proba(np.vstack(test_df["node_embeds"]))
        test_node2vec_probs = np.array([pos for neg, pos in test_node2vec_probs])

        
        scores = test_node2vec_probs
        y_test = test_df["labels"]
        
        def find_db(db):
            y_pred_test = predict(scores, db)
            recall_test = metrics.recall_score(y_test, y_pred_test, pos_label=1)
            return abs(recall_test - 0.985)
        
        auc_roc = metrics.roc_auc_score(y_test, scores)

        res = minimize_scalar(find_db)
        db = res["x"]
        y_pred_test = predict(scores, db)
        
        recall_test = metrics.recall_score(y_test, y_pred_test, pos_label=1)
        spec_test = metrics.recall_score(y_test, y_pred_test, pos_label=0)
        precision_test = calc_precision(recall_test, spec_test, neg_pos_ratio_dict[ptype])
        f1_test = calc_f1(recall_test, precision_test)
        
        metrics_dict = {
            "ptype": ptype,
            "y_test": y_test,
            "db": db,
            "scores": scores,
            "auc_roc": auc_roc,
            "recall_test": recall_test,
            "spec_test": spec_test,
            "precision_test": precision_test,
            "f1_test": f1_test
        }
        metrics_lst.append(metrics_dict)
        
        if os.path.exists(node2vec_gbdt_metrics_adjusted_results_path):
            with open(node2vec_gbdt_metrics_adjusted_results_path, "rb") as f:
                metrics_lst = pkl.load(f)
        
        metrics_lst.append(metrics_dict)
        with open(node2vec_gbdt_metrics_adjusted_results_path, "wb") as f:
            pkl.dump(metrics_lst, f)
            
        del bvc_node2vec
        gc.collect()


if __name__ == "__main__":
    main()

