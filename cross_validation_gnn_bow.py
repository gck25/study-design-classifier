import gc
import tempfile
import os.path
import pickle as pkl
import numpy as np
import random
from tqdm import tqdm
import json
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from functools import partial
from utils import get_texts_for_ptype
from sklearn.model_selection import RepeatedKFold

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
from typing import Dict

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn.aggr import PowerMeanAggregation, SoftmaxAggregation, SetTransformerAggregation, MLPAggregation, LSTMAggregation, SortAggregation, EquilibriumAggregation
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from ray import train, tune
from ray.train import Checkpoint


import datasets
from datasets import load_dataset, Dataset, load_metric

def load_data(data_dir="../data", ptype="is_rct", train_pmids=None, val_pmids=None):
    
    x_tensor_path = os.path.join(data_dir, "x_tensor_sample.pkl")
    with open(x_tensor_path, "rb") as f:
        x_tensor = pkl.load(f)
    
    pmid2node_id_path = os.path.join(data_dir, "pmid2node_id_sample.json")
    with open(pmid2node_id_path, "r") as f:
        pmid2node_id = json.load(f)
    
    PMIDS = os.path.join(data_dir, "pmids.pkl") 
    with open(PMIDS, 'rb') as f:
        pmids = pkl.load(f)
    
    edge_index_path = os.path.join(data_dir, "edge_index_sample.pkl")
    with open(edge_index_path, "rb") as f:
        edge_index = pkl.load(f)
    
    num_nodes = len(pmid2node_id)
    train_mask = [False for _ in range(num_nodes)]
    val_mask = [False for _ in range(num_nodes)]
        
    for pmid in train_pmids:
        if str(pmid) in pmid2node_id:
            node_id = pmid2node_id[str(pmid)]
            train_mask[node_id] = True
        
    for pmid in val_pmids:
        if str(pmid) in pmid2node_id:
            node_id = pmid2node_id[str(pmid)]
            val_mask[node_id] = True
        
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
        
    y = [-1 for _ in range(num_nodes)]
    pos_pmids = pmids[ptype]['pos']['train'] + pmids[ptype]['pos']['val'] + pmids[ptype]['pos']['test']
    neg_pmids = pmids[ptype]['neg']['train'] + pmids[ptype]['neg']['val'] + pmids[ptype]['neg']['test']
        
    for pmid in pos_pmids:
        if str(pmid) in pmid2node_id:
            node_id = pmid2node_id[str(pmid)]
            y[node_id] = 1
        
    for pmid in neg_pmids:
        if str(pmid) in pmid2node_id:
            node_id = pmid2node_id[str(pmid)]
            y[node_id] = 0
        
    y = torch.tensor(y, dtype=torch.long)
    edge_index = torch.tensor(edge_index)
    data = Data(x=torch.tensor(x_tensor, dtype=torch.float32), y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask)
    del y
    del x_tensor
    del edge_index
    del train_mask
    del val_mask
    
    return data




class GAT(torch.nn.Module):
    def __init__(self, hid, in_head, out_head, dropout, num_classes = 2):
        super(GAT, self).__init__()
        self.hid = hid
        self.in_head = in_head
        self.out_head = out_head
        self.dropout = dropout
        
        
        self.conv1 = GATConv(1000, self.hid, heads=self.in_head, dropout=self.dropout)
        self.conv2 = GATConv(self.hid*self.in_head, num_classes, concat=False,
                             heads=self.out_head, dropout=self.dropout)
#         self.conv = GATConv(data.num_features, num_classes, concat=True,
#                              heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class SAGENet(torch.nn.Module):
    def __init__(self, aggr="max", num_classes = 2):
        super(SAGENet, self).__init__()
        
        self.conv = SAGEConv(1000,
                             num_classes,
#                              aggr=SumAggregation()
                             aggr=aggr) # max, mean, add ...)

    def forward(self, data):
        x = self.conv(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)
    

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout, num_classes = 2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(1000, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

def train_helper(model, config, ptype, data):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model = model.to(device)
    model = model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=5e-4)
#     optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    data = data.to(device)

    for epoch in range(config["num_epochs"]):  # loop over the dataset multiple times
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    return model

def train_gat_fn(ptype):
    def train_gat(config, data):
        gat = GAT(config["hid"], config["in_head"], config["out_head"], config["dropout"], num_classes = 2)
        return train_helper(gat, config, ptype, data)
    return train_gat

def train_gcn_fn(ptype):
    def train_gcn(config, data):
        gcn = GCN(config["hidden_channels"], config["dropout"], num_classes = 2)
        return train_helper(gcn, config, ptype, data)
    return train_gcn

def train_sagenet_fn(ptype):
    def train_sagenet(config, data):
        gcn = SAGENet(config["aggr"], num_classes = 2,)
        return train_helper(gcn, config, ptype, data)
    return train_sagenet
    
def test_model(model, ptype, data):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    data = data.to(device)

    model = model.eval() 
    with torch.no_grad():
        out = model(data)
    pred = out.argmax(dim=1)
    
    metric_acc = load_metric("accuracy")
    metric_p = load_metric("precision")
    metric_r = load_metric("recall")
    metric_f = load_metric("f1")

    pred_test = pred[data.val_mask]
    labels_test = data.y[data.val_mask]
        
    accuracy_test = metric_acc.compute(predictions=pred_test, references=labels_test)['accuracy']
    precision_test = metric_p.compute(predictions=pred_test, references=labels_test)['precision']
    recall_test = metric_r.compute(predictions=pred_test, references=labels_test)['recall']
    spec_test = metrics.recall_score(labels_test.cpu().numpy(), pred_test.cpu().numpy(), pos_label=0)
    f1_test = metric_f.compute(predictions=pred_test, references=labels_test)['f1']
        
    metric_dict = {
        "accuracy_test": accuracy_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "spec_test": spec_test,
        "f1_test": f1_test
    }
    
    return metric_dict
    



def main():
    neg_to_pos_ratio = 1

    results_file = "results_titles_abstracts_gnn_bow.json"
    metrics_file = "results_crossval_gnn_bow.json"
    
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
    
    model_train_test_fns = {
        "GAT": train_gat_fn,
        "GCN": train_gcn_fn,
        "SAGENet": train_sagenet_fn
    }
    
    data_dir = "/home/k2141255/research/study_design_classifier/data"
    
    PMIDS = os.path.join(data_dir, "pmids.pkl") 
    with open(PMIDS, 'rb') as f:
        pmids = pkl.load(f)
    for results_json in results_list:
        ptype = results_json["ptype"]
        model_type = results_json["model_type"]
        config = results_json["config"]
        train_fn = model_train_test_fns[model_type]
        train_model = train_fn(ptype)
        
        random_state = 12883823
        rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=random_state)
        
        score_dict = {
            "accuracy_test": [],
            "precision_test": [],
            "recall_test": [],
            "spec_test": [],
            "f1_test": []
        }
        
        train_pmids = pmids[ptype]['pos']['train'] + pmids[ptype]['neg']['train']
        
        for fold, (train_idx, valid_idx) in enumerate(tqdm(rkf.split(train_pmids))):
            
            
            train_pmids_fold = [train_pmids[i] for i in train_idx]
            val_pmids_fold = [train_pmids[i] for i in valid_idx]
            data = load_data(data_dir=data_dir, ptype=ptype, train_pmids=train_pmids_fold, val_pmids=val_pmids_fold)
            model = train_model(config, data)
            
            metric_dict = test_model(model, ptype, data)
            
            del model
            del data
            gc.collect()
            torch.cuda.empty_cache()

            
            score_dict["accuracy_test"].append(metric_dict["accuracy_test"])
            score_dict["precision_test"].append(metric_dict["precision_test"])
            score_dict["recall_test"].append(metric_dict["recall_test"])
            score_dict["spec_test"].append(metric_dict["spec_test"])
            score_dict["f1_test"].append(metric_dict["f1_test"])

        gc.collect()
        torch.cuda.empty_cache()
        
        results_dict = {
            "ptype": ptype,
            "model_type": model_type,
            "neg_to_pos_ratio": neg_to_pos_ratio,
            "config": config,
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
        
        
if __name__ == "__main__":
    main()