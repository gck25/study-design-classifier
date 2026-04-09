import os
import copy
import ray
import json
import pickle as pkl
import pandas as pd
from tqdm import tqdm
import random as random
import numpy as np
from ray import tune
import datasets
import numpy as np
from sklearn import metrics
from datasets import load_metric

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
import ray.train.huggingface.transformers
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

ray.init(object_store_memory=10**8)

neg_to_pos_ratio = 1

results_file = "results_titles_abstracts_BERT.json"

inputs = "titles_abstracts"
results_file = "results_titles_abstracts_bert_experiments.json"

filename_pos_human = "../data/pubmed_articles_pos_human.csv"
filename_neg_human = "../data/pubmed_articles_neg_human.csv"

with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
    prop_typ_id_dict = pkl.load(f)

with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)

ptypes = prop_typ_id_dict.keys()
#     ptypes = ["is_cohort", "is_guideline", "is_consensus", "is_qual", "is_ctprotocol"]

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=420)

metric_acc = load_metric("accuracy")
metric_p = load_metric("precision")
metric_r = load_metric("recall")
metric_f = load_metric("f1")

def compute_metrics(eval_pred):
    
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)
    
    acc = metric_acc.compute(predictions=predictions, references=labels)['accuracy']
    p = metric_p.compute(predictions=predictions, references=labels)['precision']
    r = metric_r.compute(predictions=predictions, references=labels)['recall']
    f = metric_f.compute(predictions=predictions, references=labels)['f1']
    
#     tune.report(f1 = f)

    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f}

def ray_hp_space(trial):
    return {
        'num_train_epochs': tune.choice([1, 2, 3, 4, 5, 6]),
        'learning_rate': tune.choice(np.logspace(-8, -2, 20)),
        'seed': tune.choice(range(0, 1000)),
        'per_device_train_batch_size': tune.choice([8, 16])
    }

model_types = [
#     "distilbert-base-uncased", 
#     "bert-base-cased", 
#     "allenai/scibert_scivocab_uncased",
#     "dmis-lab/biobert-base-cased-v1.2",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", #PubMedBERT
]

ptypes = [
#     "is_rct",
#     "is_sr",
#     "is_casecontrol",
#     "is_cohort",
    "is_guideline"
         
         ]

for ptype in ptypes:
    pos_data_human = pd.read_csv(filename_pos_human, chunksize=100000)
    neg_data_human = pd.read_csv(filename_neg_human, chunksize=100000)

    pos_train_ids = prop_typ_id_dict[ptype]["pos"]["train"]
    neg_train_ids = prop_typ_id_dict[ptype]["neg"]["train"]

    pos_val_ids = prop_typ_id_dict[ptype]["pos"]["val"]
    neg_val_ids = prop_typ_id_dict[ptype]["neg"]["val"]

    pos_test_ids = prop_typ_id_dict[ptype]["pos"]["test"]
    neg_test_ids = prop_typ_id_dict[ptype]["neg"]["test"]

    pos_train_set = pd.DataFrame()
    pos_val_set = pd.DataFrame()
    pos_test_set = pd.DataFrame()

    for pos_chunk in tqdm(pos_data_human):

        indices = pos_chunk.index
        indices = list(indices)

        pos_train_indices = list(
            set(indices).intersection(set(pos_train_ids))
        )
        pos_val_indices = list(set(indices).intersection(set(pos_val_ids)))
        pos_test_indices = list(set(indices).intersection(set(pos_test_ids)))

        pos_train_subset = pos_chunk.loc[pos_train_indices]
        if pos_train_set.empty:
            pos_train_set = pd.DataFrame(columns=pos_train_set.columns)
        pos_train_set = pd.concat([pos_train_set, pos_train_subset])

        pos_val_subset = pos_chunk.loc[pos_val_indices]
        if pos_val_set.empty:
            pos_val_set = pd.DataFrame(columns=pos_val_set.columns)
        pos_val_set = pd.concat([pos_val_set, pos_val_subset])

        pos_test_subset = pos_chunk.loc[pos_test_indices]
        if pos_test_set.empty:
            pos_test_set = pd.DataFrame(columns=pos_test_set.columns)
        pos_test_set = pd.concat([pos_test_set, pos_test_subset])

        pos_train_set[ptype] = int(1)
        pos_val_set[ptype] = int(1)
        pos_test_set[ptype] = int(1)

    del pos_data_human
    del pos_train_subset
    del pos_chunk
    del indices
    del pos_train_ids
    
    neg_train_ids_sample = random.sample(
         list(neg_train_ids), round(len(pos_train_set) * neg_to_pos_ratio)
    )
    
    neg_train_set = pd.DataFrame()
    neg_val_set = pd.DataFrame()
    neg_test_set = pd.DataFrame()

    for neg_chunk in tqdm(neg_data_human):
        
        indices = neg_chunk.index
        indices = list(indices)
        neg_train_indices = list(
            set(indices).intersection(set(neg_train_ids_sample))            
        )
        neg_val_indices = list(set(indices).intersection(set(neg_val_ids)))
        neg_test_indices = list(set(indices).intersection(set(neg_test_ids)))
        neg_subset = neg_chunk.loc[indices]

        neg_train_subset = neg_chunk.loc[neg_train_indices]
        if neg_train_set.empty:
            neg_train_set = pd.DataFrame(columns=neg_train_set.columns)
        neg_train_set = pd.concat([neg_train_set, neg_train_subset])

        neg_val_subset = neg_chunk.loc[neg_val_indices]
        if neg_val_set.empty:
            neg_val_set = pd.DataFrame(columns=neg_val_set.columns)
        neg_val_set = pd.concat([neg_val_set, neg_val_subset])

        neg_test_subset = neg_chunk.loc[neg_test_indices]
        if neg_test_set.empty:
            neg_test_set = pd.DataFrame(columns=neg_test_set.columns)
        neg_test_set = pd.concat([neg_test_set, neg_test_subset])

        neg_train_set[ptype] = int(0)
        neg_val_set[ptype] = int(0)
        neg_test_set[ptype] = int(0)

    del neg_data_human
    del neg_subset
    del neg_chunk
    del indices

    train_data = pd.concat([neg_train_set, pos_train_set]).sample(
        frac=0.1
    )

    del neg_train_set
    del pos_train_set
    val_data = pd.concat([neg_val_set, pos_val_set]).sample(frac=0.01)
    del neg_val_set
    del pos_val_set
    test_data = pd.concat([neg_test_set, pos_test_set]).sample(frac=1)
    del neg_test_set
    del pos_test_set

    y_train = train_data[ptype]
    y_val = val_data[ptype]
    y_test = test_data[ptype]

    if inputs == "abstracts":
        X_train = train_data["abstract_plaintext"]
        X_val = val_data["abstract_plaintext"]
        X_test = test_data["abstract_plaintext"]
    elif inputs == "titles":
        X_train = train_data["title"]
        X_val = val_data["title"]
        X_test = test_data["title"]
    elif inputs == "titles_abstracts":
        X_train = pd.Series(
             [
                 "{0} {1}".format(train_title, train_abstract)
                     for train_title, train_abstract in zip(
                         train_data["title"], train_data["abstract_plaintext"]
                     )
             ]
        )
        X_val = pd.Series(
           [
               "{0} {1}".format(val_title, val_abstract)
                    for val_title, val_abstract in zip(
                        val_data["title"], val_data["abstract_plaintext"]
                    )
            ]
        )
        X_test = pd.Series(
            [
                "{0} {1}".format(test_title, test_abstract)
                    for test_title, test_abstract in zip(
                        test_data["title"], test_data["abstract_plaintext"]
                    )
            ]
        )

    del train_data
    del val_data
    del test_data

    train = pd.DataFrame()
    train['label'] = y_train
    train['text'] = list(X_train)

    val = pd.DataFrame()
    val['label'] = y_val
    val['text'] = list(X_val)

    test = pd.DataFrame()
    test['label'] = y_test
    test['text'] = list(X_test)

    train_dataset = datasets.Dataset.from_pandas(train.reset_index(drop=True))
    val_dataset = datasets.Dataset.from_pandas(val.reset_index(drop=True))
    test_dataset = datasets.Dataset.from_pandas(test.reset_index(drop=True))

    dataset = datasets.dataset_dict.DatasetDict(
        {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    )
    for model_type in model_types:
        print(ptype, model_type, "Beginning")
            
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(
                model_type, return_dict=False, force_download=True)
        
        tokenizer = AutoTokenizer.from_pretrained(model_type)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        training_args = TrainingArguments(output_dir="test_trainer", 
                                          num_train_epochs=1, 
                                          dataloader_num_workers = 1,
                                          per_device_train_batch_size=64, 
                                          logging_dir = "/home/k2141255/research/study_design_classifier/tensorboard_logs",
                                          report_to = ["tensorboard"],
                                          evaluation_strategy="steps",
                                          eval_steps=1,
                                          logging_strategy="steps",
                                          logging_steps=1)
        
        train_dataset = tokenized_datasets['train'].map(tokenize_function, batched=True)
        val_dataset = tokenized_datasets['val'].map(tokenize_function, batched=True)
        test_dataset = tokenized_datasets['test'].map(tokenize_function, batched=True)
        
        train_dataset = train_dataset.shuffle(seed=42)
        val_dataset = val_dataset.shuffle(seed=42)
        test_dataset = test_dataset.shuffle(seed=42)
        
        
        hyperopt_search = HyperOptSearch(metric="eval_f1", mode="max")
        
        

        trainer = Trainer(

        #     model=model,

            args=training_args,

            train_dataset=train_dataset,

            eval_dataset=val_dataset,

            compute_metrics=compute_metrics,

            model_init=model_init

        )
        
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="ray",
            # Choose among many libraries:
            # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
            search_alg=hyperopt_search,
            hp_space = ray_hp_space,
            n_trials=20,
            # Choose among schedulers:
            # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
        #     scheduler=ASHAScheduler(metric="f1", mode="max"),
            resources_per_trial = {'gpu':1}
        )
        
        training_args = TrainingArguments(output_dir="test_trainer", 
                                  evaluation_strategy="epoch", 
                                  num_train_epochs=best_trial[2]["num_train_epochs"], 
                                  learning_rate=best_trial[2]["learning_rate"],
                                  seed=best_trial[2]["seed"],
                                  per_device_train_batch_size=best_trial[2]["per_device_train_batch_size"],
                                  dataloader_num_workers = 1
                                 )
        
        trainer = Trainer(

#             model=model, 

            args=training_args,

            train_dataset=train_dataset,

            eval_dataset=val_dataset,

            compute_metrics=compute_metrics,

            model_init=model_init

    )

#         trainer.train()
        trainer.save_model(f'trained/{ptype}_{model_type}')
        break
    break   
        
#         outputs = trainer.predict(val_dataset)
#         predictions = np.argmax(outputs.predictions, axis=-1)
#         labels = val_dataset['label']
        
#         accuracy_val = metric_acc.compute(predictions=predictions, references=labels)['accuracy']
#         precision_val = metric_p.compute(predictions=predictions, references=labels)['precision']
#         recall_val = metric_r.compute(predictions=predictions, references=labels)['recall']
#         spec_val = metrics.recall_score(labels, predictions, pos_label=0)
#         f1_val = metric_f.compute(predictions=predictions, references=labels)['f1']

#         outputs = trainer.predict(test_dataset)
#         predictions = np.argmax(outputs.predictions, axis=-1)
#         labels = test_dataset['label']
        
#         accuracy_test = metric_acc.compute(predictions=predictions, references=labels)['accuracy']
#         precision_test = metric_p.compute(predictions=predictions, references=labels)['precision']
#         recall_test = metric_r.compute(predictions=predictions, references=labels)['recall']
#         spec_test = metrics.recall_score(labels, predictions, pos_label=0)
#         f1_test = metric_f.compute(predictions=predictions, references=labels)['f1']
        
#         results_dict = {
#                 "ptype": ptype,
#                 "model_type": model_type,
#                 "neg_to_pos_ratio": neg_to_pos_ratio,
#                 "num_train_epochs": best_trial[2]["num_train_epochs"], 
#                 "learning_rate": best_trial[2]["learning_rate"],
#                 "seed": best_trial[2]["seed"],
#                 "batch_size": best_trial[2]["per_device_train_batch_size"],
#                 "recall_val": recall_val,
#                 "spec_val": spec_val,
#                 "precision_val_even": precision_val,
#                 "f1_val_even": f1_val,
#                 "accuracy_val": accuracy_val,
#                 "recall_test": recall_test,
#                 "spec_test": spec_test,
#                 "precision_test_even": precision_test,
#                 "f1_test_even": f1_test,
#                 "accuracy_test": accuracy_test,
#         }
        
#         print(ptype, model_type, "Finished")

#         if os.path.exists(results_file):
#             with open(results_file, "r") as f_read:
#                 results_list = json.load(f_read)
#         else:
#             results_list = []

#         results_list.append(results_dict)
#         with open(results_file, "w") as f_write:
#             json.dump(results_list, f_write)

            