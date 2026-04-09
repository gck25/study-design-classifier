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
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from functools import partial
from fastnode2vec import Node2Vec, Graph
from gensim.models import Word2Vec
from custom_models.custom_model import BatchVectorizerXGBoost
from utils import get_texts_for_ptype


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
    neg_to_pos_ratio = 1

    results_file = "results_node2vec_gbdt_sample.json"
    metrics_file = "results_crossval_node2vec_gbdt_sample.json"
    MODEL_DIR = "../models"
    EMBEDDING_FILENAME = os.path.join(MODEL_DIR, 'node_model.pkl')
    PMIDS = '../data/pmids.pkl'  
    
    with open(results_file, "r") as f:
        results_list = json.load(f)
        
    with open("../data/graph_lst.pkl", "rb") as f:
        graph_lst = pkl.load(f)

    graph = Graph(graph_lst, directed=False, weighted=False)

    node2vec = Node2Vec(graph, dim=10, walk_length=100, context=10, p=2.0, q=0.5, workers=5)  # Use temp_folder for big graphs


    # Load model after Node2Vec.wv.save_word2vec_format
    model = node2vec.wv.load_word2vec_format(EMBEDDING_FILENAME)
   
    with open("../data/pmid_to_openAlex.json") as f:
        pmid_to_openAlex = json.load(f)
    
    with open(PMIDS, 'rb') as f:
        pmids = pkl.load(f)

    with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)

    ptypes = prop_typ_id_dict.keys()
    
    metrics_list = []
    for results_json in tqdm(results_list):
        ptype = results_json["ptype"]
        pos_train_embeddings, neg_train_embeddings = get_embeddings(pmids, pmid_to_openAlex, ptype, model, split='train')
        print("Positive train length", len(pos_train_embeddings))
        print("Negative train length", len(neg_train_embeddings))
        
        pos_df = pd.DataFrame({"x": pos_train_embeddings})
        pos_df['y'] = 1

        neg_df = pd.DataFrame({"x": neg_train_embeddings})
        neg_df['y'] = 0

        train_df = pd.concat([pos_df, neg_df]).sample(frac=1)
        X_train = np.array([list(el) for el in tqdm(list(train_df['x']))])
        y_train = train_df['y']
        
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
        
        
        bvc = xgb.XGBClassifier(
            
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