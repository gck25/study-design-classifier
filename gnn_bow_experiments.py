import os
import copy
import gc
import ray
import json
import pickle as pkl
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
from ray import tune
import datasets
import numpy as np
from sklearn import metrics
from datasets import load_metric

import torch
import scipy.sparse as sp
import numpy as np
import tempfile
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn.aggr import PowerMeanAggregation, SoftmaxAggregation, SetTransformerAggregation, MLPAggregation, LSTMAggregation, SortAggregation, EquilibriumAggregation
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from utils import get_texts_for_ptype

from datasets import load_dataset
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
import ray.train.huggingface.transformers
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
from typing import Dict
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler


def load_data(data_dir="../data", ptype="is_rct"):
    
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
    test_mask = [False for _ in range(num_nodes)]
        
    train_pmids = pmids[ptype]['pos']['train'] + pmids[ptype]['neg']['train']
    val_pmids = pmids[ptype]['pos']['val'] + pmids[ptype]['neg']['val']
    test_pmids = pmids[ptype]['pos']['test'] + pmids[ptype]['neg']['test']
        
    for pmid in train_pmids:
        if str(pmid) in pmid2node_id:
            node_id = pmid2node_id[str(pmid)]
            train_mask[node_id] = True
        
    for pmid in val_pmids:
        if str(pmid) in pmid2node_id:
            node_id = pmid2node_id[str(pmid)]
            val_mask[node_id] = True
                
    for pmid in test_pmids:
        if str(pmid) in pmid2node_id:
            node_id = pmid2node_id[str(pmid)]
            test_mask[node_id] = True
        
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
        
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
    data = Data(x=torch.tensor(x_tensor, dtype=torch.float32), y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    del y
    del x_tensor
    del edge_index
    del train_mask
    del val_mask
    del test_mask
    
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

def train_helper(model, config, ptype):
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
    
    data = load_data(data_dir="/home/k2141255/research/study_design_classifier/data", ptype=ptype)
    data = data.to(device)

    for epoch in range(config["num_epochs"]):  # loop over the dataset multiple times
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        
        
    model = model.eval() 
    with torch.no_grad():
        out = model(data)
    
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    pred_val = out[data.val_mask].argmax(dim=1)
    labels_val = data.y[data.val_mask]
    
        
    metric_f = load_metric("f1")
    f1_val = metric_f.compute(predictions=pred_val, references=labels_val)['f1']
    

    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
        torch.save(
            (model.state_dict(), optimizer.state_dict()), path
        )
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        train.report(
            {"loss": val_loss.cpu().numpy(), "f1": f1_val}, checkpoint=checkpoint,
        )
    del data
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Finished Training")

def train_gat_fn(ptype):
    def train_gat(config):
        gat = GAT(config["hid"], config["in_head"], config["out_head"], config["dropout"], num_classes = 2)
        train_helper(gat, config, ptype)
    return train_gat

def train_gcn_fn(ptype):
    def train_gcn(config):
        gcn = GCN(config["hidden_channels"], config["dropout"], num_classes = 2)
        train_helper(gcn, config, ptype)
    return train_gcn

def train_sagenet_fn(ptype):
    def train_sagenet(config):
        gcn = SAGENet(config["aggr"], num_classes = 2)
        train_helper(gcn, config, ptype)
    return train_sagenet
    
def test_best_helper(best_trained_model, best_result, ptype):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    data = load_data(data_dir="/home/k2141255/research/study_design_classifier/data", ptype=ptype)
    data = data.to(device)

    best_trained_model = best_trained_model.eval() 
    with torch.no_grad():
        out = best_trained_model(data)
    pred = out.argmax(dim=1)
        
    pred_val = pred[data.val_mask]
    labels_val = data.y[data.val_mask]
    
    metric_acc = load_metric("accuracy")
    metric_p = load_metric("precision")
    metric_r = load_metric("recall")
    metric_f = load_metric("f1")
        
    accuracy_val = metric_acc.compute(predictions=pred_val, references=labels_val)['accuracy']
    precision_val = metric_p.compute(predictions=pred_val, references=labels_val)['precision']
    recall_val = metric_r.compute(predictions=pred_val, references=labels_val)['recall']
    spec_val = metrics.recall_score(labels_val.cpu().numpy(), pred_val.cpu().numpy(), pos_label=0)
    f1_val = metric_f.compute(predictions=pred_val, references=labels_val)['f1']

    pred_test = pred[data.test_mask]
    labels_test = data.y[data.test_mask]
        
    accuracy_test = metric_acc.compute(predictions=pred_test, references=labels_test)['accuracy']
    precision_test = metric_p.compute(predictions=pred_test, references=labels_test)['precision']
    recall_test = metric_r.compute(predictions=pred_test, references=labels_test)['recall']
    spec_test = metrics.recall_score(labels_test.cpu().numpy(), pred_test.cpu().numpy(), pos_label=0)
    f1_test = metric_f.compute(predictions=pred_test, references=labels_test)['f1']
        
    metric_dict = {
        "accuracy_val": accuracy_val,
        "precision_val": precision_val,
        "recall_val": recall_val,
        "spec_val": spec_val,
        "f1_val": f1_val,
        "accuracy_test": accuracy_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "spec_test": spec_test,
        "f1_test": f1_test
    }
    
    return metric_dict
    

def test_best_gat(best_result, ptype):
    best_trained_model = GAT(best_result.config["hid"], best_result.config["in_head"], best_result.config["out_head"], 
                             best_result.config["dropout"], num_classes = 2)
    return test_best_helper(best_trained_model, best_result, ptype)
    
def test_best_gcn(best_result, ptype):
    best_trained_model = GCN(best_result.config["hidden_channels"], best_result.config["dropout"], num_classes = 2)
    return test_best_helper(best_trained_model, best_result, ptype)

def test_best_sagenet(best_result, ptype):
    best_trained_model = SAGENet(best_result.config["aggr"], num_classes = 2)
    return test_best_helper(best_trained_model, best_result, ptype)

config_gat = {
    "hid": tune.choice(list(range(1, 2))),
    "in_head": tune.choice(list(range(1, 2))),
    "out_head": tune.choice(list(range(1, 2))),
    "lr": tune.loguniform(1e-4, 1e-1),
    "num_epochs": tune.choice([20, 50, 100, 200, 300, 400, 500, 600]),
    "dropout": tune.uniform(0, 1)
}

config_gcn = {
    "hidden_channels": tune.choice(list(range(1, 5))),
    "lr": tune.loguniform(1e-4, 1e-1),
    "num_epochs": tune.choice([20, 50, 100, 200, 300, 400, 500, 600]),
    "dropout": tune.uniform(0, 1)
}

config_sagenet = {
    "aggr": tune.choice(["max", "mean", "add", 
#                          "median", 
                         "var", "std", 
#                          PowerMeanAggregation(p=1.0, learn=True, channels=1), 
#                          SoftmaxAggregation(t=1.0, learn=True, semi_grad=False, channels=1), 
#                         SetTransformerAggregation(channels=1, num_seed_points=1, num_encoder_blocks=1, num_decoder_blocks=1,
#                                                   heads=1, concat=True, layer_norm=False, dropout=0.0),
#                         LSTMAggregation(in_channels=1, out_channels=1),
#                         MLPAggregation(in_channels=1, out_channels=1, max_num_elements=5, num_layers=1),
#                         SortAggregation(k=10),
#                         EquilibriumAggregation(in_channels=1, out_channels=1, num_layers=[1], grad_iter=5, lamb=0.1)
                        ]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "num_epochs": tune.choice([20, 50, 100, 200, 300, 500, 600, 700, 800, 900, 1000, 1500, 2000])
}
    
def main(num_samples=2, max_num_epochs=2, gpus_per_trial=0):
    
    ray.init(object_store_memory=10**8)
    inputs = "titles_abstracts"
    results_file = "results_titles_abstracts_gnn_bow_tmp.json"

    with open("../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl", "rb") as f:
        prop_typ_id_dict = pkl.load(f)
        
    ptypes = prop_typ_id_dict.keys()
    
    model_type_config_train_test_best_models_list = [
        ("GAT", config_gat, train_gat_fn, test_best_gat),
        ("GCN", config_gcn, train_gcn_fn, test_best_gcn),
        ("SAGENet", config_sagenet, train_sagenet_fn, test_best_sagenet)
    ]
    
    with open('../data/open_alex_graph.json') as f:
        oalex_graph_dict = json.load(f)
    
    with open("../data/pmid_to_openAlex.json") as f:
        pmid_to_openAlex = json.load(f)
    
    PMIDS = '../data/pmids.pkl'    
    with open(PMIDS, 'rb') as f:
        pmids = pkl.load(f)
    
    for ptype in ptypes:
        # TODO implement below logic for each model type
        for model_type, config, train_fn, test_best_model in model_type_config_train_test_best_models_list:
            train_model = train_fn(ptype)
            scheduler = ASHAScheduler(
                max_t=max_num_epochs,
                grace_period=1,
                reduction_factor=2)

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(train_model),
                    resources={"cpu": 0, "gpu": gpus_per_trial}
                ),
                tune_config=tune.TuneConfig(
                    metric="f1",
                    mode="max",
                    scheduler=scheduler,
                    num_samples=num_samples,
                ),
                param_space=config,
            )
            results = tuner.fit()

            best_result = results.get_best_result("f1", "max")

            print("Best trial config: {}".format(best_result.config))
            print("Best trial final validation loss: {}".format(
                best_result.metrics["loss"]))
            print("Best trial final validation F1: {}".format(
                best_result.metrics["f1"]))

            metric_dict = test_best_model(best_result, ptype)

            results_dict = {
                    "ptype": ptype,
                    "model_type": model_type,
                    "config": best_result.config,
                    "recall_val": metric_dict["recall_val"],
                    "spec_val": metric_dict["spec_val"],
                    "precision_val_even": metric_dict["precision_val"],
                    "f1_val_even": metric_dict["f1_val"],
                    "accuracy_val": metric_dict["accuracy_val"],
                    "recall_test": metric_dict["recall_test"],
                    "spec_test": metric_dict["spec_test"],
                    "precision_test_even": metric_dict["precision_test"],
                    "f1_test_even": metric_dict["f1_test"],
                    "accuracy_test": metric_dict["accuracy_test"],
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
    main(num_samples=20, max_num_epochs=3000, gpus_per_trial=1)
    
    