import os
import numpy as np
import pickle as pkl
import pandas as pd
from joblib import dump, load
import matplotlib.pyplot as plt
from operator import itemgetter  
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


train_path = os.path.join(src_dir, train_file)
val_path = os.path.join(src_dir, val_file)
test_non_sys_path = os.path.join(src_dir, test_non_sys_file)
test_sys_path = os.path.join(src_dir, test_sys_file)

# hyperparameters
seed = 42
batch_size = 32

val_size = 0.2
data_type = 'abstracts'


with open(train_path, 'r') as f_train:
    train_set = pd.read_csv(f_train)
    
is_sys = lambda x: 'Systematic Review' in x
not_sys = lambda x: not 'Systematic Review' in x

train_set_sys_set_pos = train_set[list(map(is_sys, train_set['ptyp']))]
train_set_sys_neg = train_set[list(map(not_sys, train_set['ptyp']))]

train_set_sys_set_pos['is_sys'] = np.ones((len(train_set_sys_set_pos),1))
train_set_sys_neg['is_sys'] = np.zeros((len(train_set_sys_neg),1))

train_set_sys = pd.concat([train_set_sys_set_pos, train_set_sys_neg])

vectorizer = HashingVectorizer()

abstracts = vectorizer.fit_transform(train_set_sys['abstract'])

with open('vectorizer.pkl', 'wb') as f_vec:
    pkl.dump(vectorizer, f_vec)

with open('abstracts_vec_train.pkl', 'wb') as f_abs_vec_train:
    pkl.dump(abstracts_train, f_abs_vec_train)
    
sys_classifier = LogisticRegression(solver='sag')

dump(sys_classifier, 'sr_classifier.joblib') 

sys_classifier.fit(abstracts_train, train_set_sys['is_sys'])

print('train accuracy:', sys_classifier.score(abstracts_train, train_set_sys['is_sys']))
sys_train_est = sys_classifier.predict(abstracts_train)

pv, rv, fv, _ = precision_recall_fscore_support(sys_train_est, train_set_sys['is_sys'])

print('train SR precision:', pv[1])
print('train SR recall:', rv[1])
print('train SR F1 score:', fv[1])

y_prob_train = sys_classifier.predict_proba(abstracts_train)

print('train SR AUC:', roc_auc_score(test_set_sys['is_sys'], np.array(y_prob_test[:,1])))


    