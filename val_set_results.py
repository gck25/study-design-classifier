import os
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter  
from joblib import dump, load
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, plot_roc_curve
from utils import add_ptyp_vecs_calc_pos_weights, get_val_train_data
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from proposed_design_types import PROPOSED_DESIGN_TYPES

# paths
src_dir = 'data'
train_file = 'pubmed_articles_train.csv'
val_file = 'pubmed_articles_val.csv'
test_non_sys_file = 'pubmed_articles_test_non_sys.csv'
test_sys_file = 'pubmed_articles_test_sys.csv'

abstracts_vec_val_path = 'abstracts_vec_val.pkl'

sr_logistic_classifier_path = 'sr_classifier_logisticregression_saga.joblib'
# sr_logistic_classifier_path = 'sr_classifier.joblib'

train_path = os.path.join(src_dir, train_file)
val_path = os.path.join(src_dir, val_file)
test_non_sys_path = os.path.join(src_dir, test_non_sys_file)
test_sys_path = os.path.join(src_dir, test_sys_file)
# pubmed_path = os.path.join(src_dir, pubmed_file)

emb_path_train = os.path.join(src_dir, 'sentBERT_embeddings_train.pkl')

emb_path_val = os.path.join(src_dir, 'sentBERT_embeddings_val.pkl')

# hyperparameters
seed = 42
batch_size = 32

val_size = 0.2
data_type = 'abstracts'


print('getting vectorizer')
with open('vectorizer.pkl', 'rb') as f_vec:
    vectorizer = pkl.load(f_vec)

print('loading val set abstracts')
with open(val_path, 'r') as f_val:
    val_set = pd.read_csv(f_val)

print('loading val abstracts vectors')
with open(abstracts_vec_val_path, 'rb') as f_abs_vec_val:
    abstracts_val = pkl.load(f_abs_vec_val)

# print('vectorizing val abstracts')
# abstracts_val = vectorizer.transform(val_set['abstract'])

# print('saving vectorized val abstracts')
#
# with open(abstracts_vec_val_path, 'wb') as f_abs_vec_val:
#     pkl.dump(abstracts_val, f_abs_vec_val)
    
is_sys = lambda x: 'Systematic Review' in x
not_sys = lambda x: not 'Systematic Review' in x

print('generating labels for val set')

val_set_sys_pos = val_set[list(map(is_sys, val_set['ptyp']))]
val_set_sys_neg = val_set[list(map(not_sys, val_set['ptyp']))]

val_set_sys_pos['is_sys'] = np.ones((len(val_set_sys_pos),1))
val_set_sys_neg['is_sys'] = np.zeros((len(val_set_sys_neg),1))

print('val sys pos:', len(val_set_sys_pos))
print('val sys neg:', len(val_set_sys_neg))

val_set_sys = pd.concat([val_set_sys_pos, val_set_sys_neg])

print('loading logistic regression SR classifier')
sys_classifier = load(sr_logistic_classifier_path)

print('val accuracy:', sys_classifier.score(abstracts_val, val_set_sys['is_sys']))
sys_val_est = sys_classifier.predict(abstracts_val)

print('calculating P, R, F1')
pv, rv, fv, _ = precision_recall_fscore_support(sys_val_est, val_set_sys['is_sys'])

print('val SR precision:', pv[1])
print('val SR recall:', rv[1])
print('val SR F1 score:', fv[1])

y_prob_val = sys_classifier.predict_proba(abstracts_val)

print('val SR AUC:', roc_auc_score(val_set_sys['is_sys'], np.array(y_prob_val[:,1])))
