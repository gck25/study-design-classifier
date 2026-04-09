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

study_design = 'is_guideline'

filename_pos_human = '/home/k2141255/study_design_classifier/data/pubmed_articles_pos_human.csv'

filename_neg_human = '/home/k2141255/study_design_classifier/data/pubmed_articles_neg_human.csv'

ids_file = '/home/k2141255/study_design_classifier/data/cleaned_pos_neg_train_val_test_dataset_ids.pkl'


with open(ids_file, 'rb') as f:
    prop_typ_id_dict= pkl.load(f)

print('loading positive data')
pos_data_human = pd.read_csv(
         filename_pos_human,
)
    
print('setting positive rows') 
pos_ids = prop_typ_id_dict[study_design]['pos']
pos_train_set = pos_data_human.loc[pos_ids['train']]
pos_test_set = pos_data_human.loc[pos_ids['test']]
del pos_data_human



pos_train_set[study_design] = int(1)
# pos_val_set = pos_data_human.loc[pos_ids['val']]
# pos_val_set[study_design] = int(1)

pos_test_set[study_design] = int(1)


# pos_set  = pos_data_human.loc[pos_ids]
# pos_set[study_design] = int(1)

print('loading negative data')
neg_data_human = pd.read_csv(filename_neg_human)

print('setting negative rows')
neg_ids = prop_typ_id_dict[study_design]['neg']
neg_train_set = neg_data_human.loc[neg_ids['train']]
neg_test_set = neg_data_human.loc[neg_ids['test']]
del neg_data_human

neg_train_set[study_design] = int(0)
# neg_val_set = neg_data_human.loc[neg_ids['val']]
# neg_val_set[study_design] = int(0)
neg_test_set[study_design] = int(0)
# neg_ids = prop_typ_id_dict[study_design]['neg']
# neg_set  = neg_data_human.loc[neg_ids]
# neg_set[study_design] = int(0)



print('sample negative train data')
neg_train_set_sample = neg_train_set.sample(n=len(pos_train_set)*neg_to_pos_ratio)
del neg_train_set

print('concatenate train data')
data_train = pd.concat([neg_train_set_sample, pos_train_set]).sample(frac=1)
del neg_train_set_sample
del pos_train_set

print('train inputs and labels')
labels_train = data_train[study_design]
inputs_train = data_train['abstract_plaintext']
del data_train

# print('concatenate val data')
# data_val = pd.concat([neg_val_set, pos_val_set])
# del neg_val_set
# del pos_val_set

# print('val inputs and labels')
# labels_val = data_val[study_design]
# inputs_val = data_val['abstract_plaintext']
# del data_val

print('concatenate test data')
data_test = pd.concat([neg_test_set, pos_test_set])
del neg_test_set
del pos_test_set

print('test inputs and labels')
labels_test = data_test[study_design]
inputs_test = data_test['abstract_plaintext']
del data_test

# X_train, X_test, y_train, y_test =  train_test_split(inputs, labels, test_size=0.2)
# del inputs
# del labels

# X_val, X_test, y_val, y_test =  train_test_split(X_test_val, y_test_val, test_size=0.5)

print('vectorize train data')
vec = HashingVectorizer(ngram_range=(1, 4), stop_words='english')
inputs_train_trans = vec.fit_transform(inputs_train.values.astype('U'))
del inputs_train

grid = {"loss": ["log"],
        "penalty": ["l2"],
        "alpha": np.logspace(-10, -5, 20)}

# print('grid search of LR')
# clf = GridSearchCV(SGDClassifier(), grid, n_jobs=4, scoring='roc_auc',verbose=100)

clf = SGDClassifier(loss="log", penalty="l2", alpha=0.0001)
clf.fit(inputs_train_trans, labels_train)

with open('pubmed_is_ct_trained_lr.pkl', 'wb') as f:
    pkl.dump(clf, f)
    
# with open('pubmed_{}_trained_lr.pkl'.format(study_design), 'wb') as f:
#     pkl.dump(clf.best_estimator_, f)



print('vectorize test data')
inputs_test_trans = vec.transform(inputs_test.values.astype('U'))
del inputs_test

print('perform inference on test data')
labels_pred = clf.predict(inputs_test_trans)

metrics.plot_roc_curve(clf, inputs_test_trans, labels_test) 
plt.show()

print('precision',metrics.precision_score(labels_pred, labels_test))
print('recall',metrics.recall_score(labels_pred, labels_test))
print('f1',metrics.f1_score(labels_pred, labels_test))
print('accuracy',metrics.accuracy_score(labels_pred, labels_test))
print('AUC ROC',metrics.roc_auc_score(labels_pred, labels_test))
print(len(labels_test))
