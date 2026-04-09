import gc
import os.path
import pandas as pd
import pickle as pkl
import numpy as np
import random
from tqdm import tqdm
import json
import torch
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from functools import partial
from utils import get_texts_for_ptype
from sklearn.model_selection import RepeatedKFold

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import datasets
from datasets import load_dataset, Dataset, load_metric


def main():
    neg_to_pos_ratio = 1

    results_file = "results_titles_abstracts_bert_experiments.json"
    metrics_file = "results_crossval_bert.json"
    
    with open(results_file, "r") as f:
        results_list = json.load(f)
    
    inputs = "titles_abstracts"

    filename_pos_human = "../data/pubmed_articles_pos_human.csv"
    filename_neg_human = "../data/pubmed_articles_neg_human.csv"

    with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)

    ptypes = prop_typ_id_dict.keys()
    
    metrics_list = []
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
        return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f}
    
    for results_json in results_list[1:]:
        ptype = results_json["ptype"]
        model_type = results_json["model_type"]
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(model_type, return_dict=False, force_download=True)
        X_train, X_val, X_test, y_train, y_val, y_test = get_texts_for_ptype(ptype, inputs="titles_abstracts", neg_to_pos_ratio=neg_to_pos_ratio)
        
        train = pd.DataFrame()
        train['label'] = y_train
        train['text'] = list(X_train)
        
        

        train_dataset = datasets.Dataset.from_pandas(train.reset_index(drop=True))
        
        dataset = datasets.dataset_dict.DatasetDict(
            {
                'train': train_dataset,
            }
        )
        
        random_state = 12883823
        rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=random_state)
        
        score_dict = {
            "accuracy_test": [],
            "precision_test": [],
            "recall_test": [],
            "spec_test": [],
            "f1_test": []
        }
        
        
        for fold, (train_idx, valid_idx) in enumerate(tqdm(rkf.split(X_train))):
            
            
            tokenizer = AutoTokenizer.from_pretrained(model_type)
            def tokenize_function(examples):
                return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=420)

            tokenized_datasets = dataset.map(tokenize_function, batched=True)

            
            val_dataset = pd.DataFrame(
                {
                    "label": pd.Series(dataset['train']['label'])[valid_idx],
                   "text": pd.Series(dataset['train']['text'])[valid_idx]
                }
            )
            val_dataset = Dataset.from_pandas(val_dataset)
            
            train_dataset = pd.DataFrame(
                {
                    "label": pd.Series(dataset['train']['label'])[train_idx],
                   "text": pd.Series(dataset['train']['text'])[train_idx]
                }
            )
            train_dataset = Dataset.from_pandas(train_dataset)
            val_dataset = val_dataset.shuffle(seed=42)
            train_dataset = train_dataset.shuffle(seed=42)
            
            train_dataset_tok = train_dataset.map(tokenize_function, batched=True)
            val_dataset_tok = val_dataset.map(tokenize_function, batched=True)
            
            training_args = TrainingArguments(output_dir="test_trainer", 
                                              evaluation_strategy="epoch", 
                                              num_train_epochs=results_json["num_train_epochs"], 
                                              learning_rate=results_json["learning_rate"],
                                              seed=results_json["seed"],
                                              per_device_train_batch_size=results_json["batch_size"],
                                              dataloader_num_workers = 1
                                             )

            trainer = Trainer(
                args=training_args,
                train_dataset=train_dataset_tok,
                eval_dataset=val_dataset_tok,
                compute_metrics=compute_metrics,
                model_init=model_init
            )

            trainer.train()
            outputs = trainer.predict(val_dataset_tok)
            predictions = np.argmax(outputs.predictions, axis=-1)
            labels = val_dataset['label']
            
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

            accuracy_val = metric_acc.compute(predictions=predictions, references=labels)['accuracy']
            score_dict["accuracy_test"].append(accuracy_val)

            precision_val = metric_p.compute(predictions=predictions, references=labels)['precision']
            score_dict["precision_test"].append(precision_val)

            recall_val = metric_r.compute(predictions=predictions, references=labels)['recall']
            score_dict["recall_test"].append(recall_val)

            spec_val = metrics.recall_score(labels, predictions, pos_label=0)
            score_dict["spec_test"].append(spec_val)

            f1_val = metric_f.compute(predictions=predictions, references=labels)['f1']
            score_dict["f1_test"].append(f1_val)

        gc.collect()
        torch.cuda.empty_cache()
        
        results_dict = {
            "ptype": ptype,
            "model_type": model_type,
            "neg_to_pos_ratio": neg_to_pos_ratio,
            "num_train_epochs": results_json["num_train_epochs"], 
            "learning_rate": results_json["learning_rate"],
            "seed": results_json["seed"],
            "batch_size": results_json["batch_size"],
            "score_dict": score_dict
        }

        print(ptype, model_type, "Finished")

        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f_read:
                results_list = json.load(f_read)
        else:
            results_list = []

        results_list.append(results_dict)
        with open(metrics_file, "w") as f_write:
            json.dump(results_list, f_write)

                
        del X_train, X_val, X_test, y_train, y_val, y_test
        
        
if __name__ == "__main__":
    main()