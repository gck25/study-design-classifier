import os
import gc
import torch
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
from torch.utils.data import DataLoader
from utils import get_texts_for_ptype

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import load_dataset
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
import ray.train.huggingface.transformers
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def mean_pooling(token_embeddings, mask):
    """
    Applies mean pooling to token embeddings to create sentence embeddings.

    Args:
        token_embeddings: torch.tensor
            token embeddings created using LLM
        mask: torch.tensor
            mask to apply to embeddings

    Returns:
        sentence_embeddings: torch.tensor
            sentence embeddings for information retrieval
    """
    # apply mask to token embeddings
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)

    # calculate mean of token embeddings to create sentence embeddings
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def create_dataset_tokenized(dataset, tokenizer, device):
    """
    Tokenizes input dataset and allocates data to CPU or GPU.

    Args:
        dataset: list(str)
            question ar answer dataset
        tokenizer: AutoTokenizer
            LLM tokenizer to be applied to dataset
        device: torch.device
            CPU or GPU device

    Returns:
        dataset_tokenized: dict(str) = torch.tensor
            tokenized dataset allocated to device
    """
    # tokenize dataset
    dataset_tokenized = tokenizer(dataset, padding=True, truncation=True, max_length=512)

    # convert dataset to torch dataset and allocate to device
    dataset_tokenized = {
        'input_ids': torch.tensor(dataset_tokenized['input_ids']).to(device),
#         'token_type_ids': torch.tensor(dataset_tokenized['token_type_ids']).to(device),
        'attention_mask': torch.tensor(dataset_tokenized['attention_mask']).to(device),
    }

    return dataset_tokenized

def main():
    with open("../data/x_texts.pkl", 'rb') as f:
        x_texts = pkl.load(f)
        
    model_name_types_tuples = [
#         ('BioBERT', 'dmis-lab/biobert-base-cased-v1.2'),
#         ('PubMedBERT', 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'),
        ('DistilBERT', 'distilbert-base-uncased')
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size_outer = 160000
    batch_size_inner = 16

    num_iter_outer = len(x_texts)//batch_size_outer
    num_iter_inner = batch_size_outer//batch_size_inner
    
    for name, model_type in tqdm(model_name_types_tuples):
        counter = 0
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        print('tokenizer')
        model = AutoModel.from_pretrained(model_type).to(device)
        print('model')
        for i in tqdm(range(num_iter_outer)):
            x_texts_tokenized = create_dataset_tokenized(list(x_texts[i*batch_size_outer:(i+1)*batch_size_outer]), tokenizer, device)
            x_texts_loader = DataLoader(list(zip(x_texts_tokenized['input_ids'], 
#                                                  x_texts_tokenized['token_type_ids'],
                                                 x_texts_tokenized['attention_mask'])), batch_size=batch_size_inner)
#             for input_ids, token_type_ids, attention_mask in tqdm(x_texts_loader):
            for input_ids, attention_mask in tqdm(x_texts_loader):
                batch_x_tok = {
                    'input_ids': input_ids,
#                     'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask
                }
                x_outputs = model(**batch_x_tok)

                x_embedding = mean_pooling(x_outputs[0], mask=attention_mask).cpu().detach().numpy()
                with open(f'../data/x_bert/x_{name}_{counter}_embeddings.pkl', 'wb') as f:
                    pkl.dump(x_embedding, f)
                counter +=1
        x_texts_tokenized = create_dataset_tokenized(list(x_texts[num_iter_outer*batch_size_outer:]), tokenizer, device)

        x_texts_loader = DataLoader(list(zip(x_texts_tokenized['input_ids'], 
#                                              x_texts_tokenized['token_type_ids'],
                                                 x_texts_tokenized['attention_mask'])), batch_size=batch_size_inner)

#         for input_ids, token_type_ids, attention_mask in tqdm(x_texts_loader):
        for input_ids, attention_mask in tqdm(x_texts_loader):
            batch_x_tok = {
                'input_ids': input_ids,
#                 'token_type_ids': token_type_ids,
                'attention_mask': attention_mask
            }
            x_outputs = model(**batch_x_tok)
            x_embedding = mean_pooling(x_outputs[0], mask=attention_mask).cpu().detach().numpy()
            with open(f'../data/x_bert/x_{name}_{counter}_embeddings.pkl', 'wb') as f:
                pkl.dump(x_embedding, f)
                print('save')
            counter +=1

        del model
        gc.collect()
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()
    
    