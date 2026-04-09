import os.path
import gc
import pandas as pd
import pickle as pkl
import torch
import numpy as np
import random
from tqdm import tqdm
import json
from sklearn import metrics
from utils import get_texts_for_ptype
from sklearn import metrics
from scipy.optimize import minimize_scalar
import datasets
from datasets import load_metric

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

neg_to_pos_ratio = 1
results_file = "results_titles_abstracts_bert_experiments.json"
    
with open(results_file, "r") as f:
    results_list = json.load(f)
    
inputs = "titles_abstracts"

with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
    prop_typ_id_dict = pkl.load(f)

ptypes = prop_typ_id_dict.keys()
    
metrics_list = []
metric_acc = load_metric("accuracy")
metric_p = load_metric("precision")
metric_r = load_metric("recall")
metric_f = load_metric("f1")

neg_to_pos_ratio = 1
results_file = "results_titles_abstracts_bert_experiments.json"
    
with open(results_file, "r") as f:
    results_list = json.load(f)
    
inputs = "titles_abstracts"

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

ptype = "is_guideline"
model_type = "bert-base-cased"
saved_model = "is_guideline_bert-base-cased"

results_json = [json for json in results_list if json['ptype'] == ptype and json['model_type'] == model_type][0]

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_type, return_dict=False, force_download=True)

X_train, X_val, X_test, y_train, y_val, y_test = get_texts_for_ptype(ptype, inputs="titles_abstracts", neg_to_pos_ratio=neg_to_pos_ratio)
    
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
    
tokenizer = AutoTokenizer.from_pretrained(model_type)
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=420)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
train_dataset = tokenized_datasets['train'].map(tokenize_function, batched=True)
val_dataset = tokenized_datasets['val'].map(tokenize_function, batched=True)
test_dataset = tokenized_datasets['test'].map(tokenize_function, batched=True)
        
train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)
test_dataset = test_dataset.shuffle(seed=42)

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
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    model_init=model_init
)

trainer.train()
trainer.save_model(f'trained/{ptype}_{model_type}_rechecked')

def get_scores(predictions):
    return [np.exp(p[1])/(np.exp(p[0]) + np.exp(p[1])) for p in predictions]

def find_db(db):
    y_pred_test = predict(scores, db)
    recall_test = metrics.recall_score(y_test, y_pred_test, pos_label=1)
    return abs(recall_test - 0.985)

def predict(scores, db=0):
    return [int(x>=db) for x in scores]

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

outputs = model(test_dataset)
predictions = outputs.predictions
scores = get_scores(predictions)
    
auc_roc = metrics.roc_auc_score(y_test, scores)
    
res = minimize_scalar(find_db)
db = res["x"]
y_pred_test = predict(scores, db)
    
recall_test = metrics.recall_score(y_test, y_pred_test, pos_label=1)
spec_test = metrics.recall_score(y_test, y_pred_test, pos_label=0)
precision_test = calc_precision(recall_test, spec_test, neg_pos_ratio_dict[ptype])
f1_test = calc_f1(recall_test, precision_test)

print("Recall:", recall_test)
print("Specificity:", spec_test)
print("Precision:", precision_test)
print("F1:", f1_test)
