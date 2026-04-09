import random
import pickle as pkl
import pandas as pd
from tqdm import tqdm

def get_texts_citations_labels(data_set):
    texts = []
    node_embeds = []
    labels = []
    for pmid in data_set:
        texts.append(data_set[pmid]["X_text"])
        node_embeds.append(data_set[pmid]["X_graph"])
        labels.append(data_set[pmid]["y"])

    return pd.DataFrame(
        {
            "texts": texts,
            "node_embeds": node_embeds,
            "labels": labels
        }
    )

def get_embeddings(pmids, pmid_to_openAlex, study_design, model, split='train'):
    
    pos_pmids = pmids[study_design]['pos'][split]
    pos_embeddings = {}

    for pmid in tqdm(pos_pmids):
        try:
            openAlexID = pmid_to_openAlex[str(pmid)]
            embedding = model.get_vector(openAlexID)
            pos_embeddings[pmid] = embedding
        except:
            continue

    neg_pmids = pmids[study_design]['neg'][split]
    neg_embeddings = {}
    for pmid in tqdm(neg_pmids):
        try:
            openAlexID = pmid_to_openAlex[str(pmid)]
            embedding = model.get_vector(openAlexID)
            neg_embeddings[pmid] = embedding
        except:
            continue
    return pos_embeddings, neg_embeddings

def get_texts_for_ptype_pmid(ptype, inputs="titles_abstracts", neg_to_pos_ratio=1, reset_index=True):
    inputs = "titles_abstracts"

    filename_pos_human = "../data/pubmed_articles_pos_human.csv"
    filename_neg_human = "../data/pubmed_articles_neg_human.csv"

    with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)

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
        list(sr_neg_train_ids), round(len(sr_pos_train_set) * neg_to_pos_ratio)
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
    if inputs == "titles_abstracts":
        train_set = {
            pmid: {"X": f"{train_title} {train_abstract}", "y": y}
            for pmid, train_title, train_abstract, y in zip(
                sr_train_data['pmid'], sr_train_data["title"], sr_train_data["abstract_plaintext"], sr_train_data[ptype]
            )
            
        }
        val_set = {
            pmid: {"X": f"{val_title} {val_abstract}", "y": y}
            for pmid, val_title, val_abstract, y in zip(
                sr_val_data['pmid'], sr_val_data["title"], sr_val_data["abstract_plaintext"], sr_train_data[ptype]
            )
        }
        test_set = {
            pmid: {"X": f"{test_title} {test_abstract}", "y": y}
            for pmid, test_title, test_abstract, y in zip(
                sr_test_data['pmid'], sr_test_data["title"], sr_test_data["abstract_plaintext"], sr_train_data[ptype]
            )
        }
    del sr_train_data
    del sr_val_data
    del sr_test_data
    
    return train_set, val_set, test_set

def inner_crossval(X, y, results_json, metric, K):
    e_in = []
    kf = KFold(n_splits=K)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_subsubtrain = X.reset_index(drop=True)[list(train_index)]
        y_subsubtrain = y.reset_index(drop=True)[list(train_index)]
            
        X_subsubtest = X.reset_index(drop=True)[list(test_index)]
        y_subsubtest = y.reset_index(drop=True)[list(test_index)]
        bvc_inner = BatchVectorizerClassifier(
                    loss=results_json["model_type"],
                    penalty=results_json["best_penalty"],
                    alpha=results_json["best_alpha"],
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
        bvc_inner.fit(X_subsubtrain, y_subsubtrain)
        y_pred = bvc_inner.predict_proba(X_subsubtest)
        e_temp = [metric(yp, y) for yp,y in zip(y_pred, y_subsubtest)]
        e_in.append(e_temp)
    return e_in
        

def nested_crossval(X, y, results_json, metric, K=10, R=10):
    es = []
    a_list = []
    b_list = []
    
    for _ in range(R):
        kf = KFold(n_splits=K)
        for i, (train_index, test_index) in enumerate(tqdm(kf.split(X))):
            X_subtrain = X[list(train_index)]
            y_subtrain = y[list(train_index)]
            
            X_subtest = X[list(test_index)]
            y_subtest = y[list(test_index)]
            e_in = inner_crossval(X_subtrain, y_subtrain, results_json, loss, K-1)
            bvc_outer = BatchVectorizerClassifier(
                    loss=results_json["model_type"],
                    penalty=results_json["best_penalty"],
                    alpha=results_json["best_alpha"],
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
            bvc_outer.fit(X_subtrain, y_subtrain)
            y_pred = bvc_outer.predict_proba(X_subtest)
            e_out = [metric(yp, y) for yp,y in zip(y_pred, y_subtest)]
            a_list.append((np.mean(e_in) - np.mean(e_out))**2)
            b_list.append(np.var(e_out)/len(X_subtest))
            es.append(e_in)
    mse = np.mean(a_list) - np.mean(b_list)
    error = np.mean(es)
    return error, mse

def get_texts_for_ptype(ptype, inputs="titles_abstracts", neg_to_pos_ratio=1, reset_index=True):
    inputs = "titles_abstracts"

    filename_pos_human = "../data/pubmed_articles_pos_human.csv"
    filename_neg_human = "../data/pubmed_articles_neg_human.csv"

    with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)

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
        list(sr_neg_train_ids), round(len(sr_pos_train_set) * neg_to_pos_ratio)
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

    y_train = sr_train_data[ptype].reset_index(drop=True)
    y_val = sr_val_data[ptype].reset_index(drop=True)
    y_test = sr_test_data[ptype].reset_index(drop=True)
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
        if reset_index:
            X_train = X_train.reset_index(drop=True)
            X_val = X_val.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

    del sr_train_data
    del sr_val_data
    del sr_test_data
    
    return X_train, X_val, X_test, y_train, y_val, y_test
           