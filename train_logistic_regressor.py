import pandas as pd
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline

neg_to_pos_ratio = 1
# seed = 42

study_design = 'is_ct'

filename_pos_human = 'pubmed_articles_pos_human.csv'

filename_neg_human = 'pubmed_articles_neg_human.csv'

print('loading positive data')
pos_data_human = pd.read_csv(
         filename_pos_human,
)


with open('pos_neg_dataset_ids.pkl', 'rb') as f:
    prop_typ_id_dict= pkl.load(f)

print('setting positive rows') 
pos_ids = prop_typ_id_dict[study_design]['pos']
pos_set  = pos_data_human.loc[pos_ids]
pos_set[study_design] = int(1)
del pos_data_human

print('loading negative data')
neg_data_human = pd.read_csv(filename_neg_human)

print('setting negative rows')
neg_ids = prop_typ_id_dict[study_design]['neg']
neg_set  = neg_data_human.loc[neg_ids]
neg_set[study_design] = int(0)
del neg_data_human


print('sample negative data')
neg_set_sample = neg_set.sample(n=len(pos_set)*neg_to_pos_ratio)
del neg_set

print('concatenate data')
data = pd.concat([neg_set_sample, pos_set]).sample(frac=1)
del neg_set_sample
del pos_set

print('inputs and labels')
labels = data[study_design]
inputs = data['abstract_plaintext']
del data

X_train, X_test, y_train, y_test =  train_test_split(inputs, labels, test_size=0.2)
del inputs
del labels

# X_val, X_test, y_val, y_test =  train_test_split(X_test_val, y_test_val, test_size=0.5)

print('vectorize train data')
vec = HashingVectorizer(ngram_range=(1, 4), stop_words='english')
X_train_trans = vec.fit_transform(X_train.values.astype('U'))

grid = {"loss": ["log"],
        "penalty": ["l2"],
        "alpha": np.logspace(-10, -5, 20)}

# print('grid search of LR')
# clf = GridSearchCV(SGDClassifier(), grid, n_jobs=4, scoring='roc_auc',verbose=100)

clf = SGDClassifier(loss="log", penalty="l2", alpha=0.0001)
clf.fit(X_train_trans, y_train)

with open('pubmed_is_ct_trained_lr.pkl', 'wb') as f:
    pkl.dump(clf, f)
    
# with open('pubmed_{}_trained_lr.pkl'.format(study_design), 'wb') as f:
#     pkl.dump(clf.best_estimator_, f)

print('vectorize test data')
X_test_trans = vec.transform(X_test.values.astype('U'))

print('perform inference on test data')
y_pred = clf.predict(X_test_trans)

metrics.plot_roc_curve(clf, X_test_trans, y_test) 
plt.show()

print('precision',metrics.precision_score(y_pred, y_test))
print('recall',metrics.recall_score(y_pred, y_test))
print('f1',metrics.f1_score(y_pred, y_test))
print('accuracy',metrics.accuracy_score(y_pred, y_test))
print('AUC ROC',metrics.roc_auc_score(y_pred, y_test))
print(len(y_test))
