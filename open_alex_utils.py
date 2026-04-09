import os
import json
import copy
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from random import shuffle

def get_pos_citation_dataset(pos_pmids, src_dir = 'open_alex_citations'):
    pos_set = []
    feature_lst = []
    for pos_pmid in tqdm(pos_pmids):
        file_name = 'open_alex_pubmed_pmid' + str(pos_pmid) + '.json'
        pos_path = os.path.join(src_dir, file_name)
        if os.path.exists(pos_path):
            if os.path.getsize(pos_path) == 0:
                continue
            else:
                with open(pos_path, 'r') as f:
                    features = json.load(f)
                    pos_set.append(features)
                    feature_lst.extend(features)
    return pos_set, feature_lst

def get_neg_citation_dataset(neg_pmids, len_pos_set = float('inf'), src_dir = 'open_alex_citations'):
    neg_set = []
    feature_lst = []
    for neg_pmid in tqdm(neg_pmids):
        if len(neg_set) == len_pos_set:
            break
        neg_file_name = 'open_alex_pubmed_pmid' + str(neg_pmid) + '.json'
        neg_path = os.path.join('open_alex_citations', neg_file_name)
        if os.path.exists(neg_path):
            if os.path.getsize(neg_path) == 0:
                continue
            else:
                with open(neg_path, 'r') as f:
                    features = json.load(f)
                    neg_set.append(features)
                    feature_lst.extend(features)
    return neg_set, feature_lst

def extract_bow_feats(dataset, feat_lst, lim_len = -1):
    bow_feats = np.array([])
    start = True
    for row in tqdm(dataset[:lim_len]):
        vec = np.zeros(len(feat_lst), dtype = int)
        for feat in row:
            if feat in feat_lst:
                index = feat_lst.index(feat)
                vec[index] = 1
        if start:
            bow_feats = csr_matrix(vec)
            start = False
        else:
            bow_feats = np.vstack([bow_feats, csr_matrix(vec)])
    return bow_feats

def get_bow_inputs_labels(bow_pos_set, bow_neg_set, is_shuffle = True):
    inputs = np.vstack([bow_pos_set, bow_neg_set])
    inputs = np.array([el[0].todense() for el in inputs], dtype='uint8')
    labels = np.array([1]*len(bow_pos_set)+ [0]*len(bow_neg_set))
    inputs = np.squeeze(inputs)
    if is_shuffle:
        index_shuf = list(range(len(labels)))
        shuffle(index_shuf)
        n = 100
        labels_shuffled = copy.copy(labels)
        inputs_shuffled = copy.copy(inputs)
        for i in tqdm(range(len(index_shuf)//n)):
            labels_shuffled[i*n: (i+1)*n] = labels[index_shuf[i*n: (i+1)*n]]
            inputs_shuffled[i*n: (i+1)*n] = inputs[index_shuf[i*n: (i+1)*n]]

        labels_shuffled[(i+1)*n:] = labels[index_shuf[(i+1)*n:]]
        inputs_shuffled[(i+1)*n:] = inputs[index_shuf[(i+1)*n:]]
        return inputs_shuffled, labels_shuffled
    return inputs, labels