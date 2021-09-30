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
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
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
sr_logistic_classifier_path = 'sr_classifier_logisticregression_sag_equal_split.joblib'


train_path = os.path.join(src_dir, train_file)
val_path = os.path.join(src_dir, val_file)
test_non_sys_path = os.path.join(src_dir, test_non_sys_file)
test_sys_path = os.path.join(src_dir, test_sys_file)

# hyperparameters
seed = 42
batch_size = 32
solver = 'sag'

val_size = 0.2
data_type = 'abstracts'


print('loading train set')
with open(train_path, 'r') as f_train:
    train_set = pd.read_csv(f_train)

with open('abstracts_vec_train.pkl', 'rb') as f_abs_vec_train:
    abstracts_train = pkl.load(f_abs_vec_train)

train_set['BoW abstracts'] = np.array(abstracts_train)

is_sys = lambda x: 'Systematic Review' in x
not_sys = lambda x: not 'Systematic Review' in x

print('assigning labels for SR classifier')

train_set_sys_pos = train_set[list(map(is_sys, train_set['ptyp']))]
train_set_sys_neg = train_set[list(map(not_sys, train_set['ptyp']))]

train_set_sys_neg = train_set_sys_neg.sample(n=len(train_set_sys_pos), replace = False)

train_set_sys_pos['is_sys'] = np.ones((len(train_set_sys_pos),1))
train_set_sys_neg['is_sys'] = np.zeros((len(train_set_sys_neg),1))

train_set_sys = pd.concat([train_set_sys_pos, train_set_sys_neg])


# with open('vectorizer.pkl', 'rb') as f_vec:
#     vectorizer = pkl.load(f_vec)





# TFIDF vectorizer
# hashing = HashingVectorizer(non_negative=True, norm=None)
# tfidf = TfidfTransformer()
# hashing_tfidf = Pipeline([("hashing", hashing), ("tidf", tfidf)])
# hashing_tfidf = Pipeline([("hashing", hashing), ("tidf", tfidf)])

# vectorizer = HashingVectorizer()

# print('fitting and transforming vectorizer')

# abstracts_train = vectorizer.fit_transform(train_set_sys['abstract'])

# print('saving vectorizer')
# with open('vectorizer.pkl', 'wb') as f_vec:
#     pkl.dump(vectorizer, f_vec)
#
# with open('abstracts_vec_train.pkl', 'wb') as f_abs_vec_train:
#     pkl.dump(abstracts_train, f_abs_vec_train)
    
print('training logistic regression classifier')
    
sys_classifier = LogisticRegression(solver=solver, random_state = seed)



sys_classifier.fit(np.array(train_set['BoW abstracts']), train_set_sys['is_sys'])

print('saving logistic regression classifier')
dump(sys_classifier, sr_logistic_classifier_path)

print('train accuracy:', sys_classifier.score(np.array(train_set['BoW abstracts']), train_set_sys['is_sys']))
sys_train_est = sys_classifier.predict(np.array(train_set['BoW abstracts']))

pv, rv, fv, _ = precision_recall_fscore_support(sys_train_est, train_set_sys['is_sys'])

print('train SR precision:', pv[1])
print('train SR recall:', rv[1])
print('train SR F1 score:', fv[1])

y_prob_train = sys_classifier.predict_proba(abstracts_train)

print('train SR AUC:', roc_auc_score(train_set_sys['is_sys'], np.array(y_prob_train[:,1])))


    