import os
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from utils import add_ptyp_vecs_calc_pos_weights, get_val_train_data
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay

# paths
src_dir = '.'
pubmed_file = 'pubmed_articles.csv'

pubmed_path = os.path.join(src_dir, pubmed_file)
emb_path_train = os.path.join(src_dir, 'sentBERT_embeddings_train.pkl')

emb_path_val = os.path.join(src_dir, 'sentBERT_embeddings_train.pkl')

# hyperparameters
seed = 42
batch_size = 32

val_size = 0.2
data_type = 'abstracts'


with open(pubmed_path, 'r') as f:
    data = pd.read_csv(f)

pos_weight_class, y_labels = add_ptyp_vecs_calc_pos_weights(data)
data_train, data_val = get_val_train_data(data, y_labels, data_type, val_size, seed)

x_train, y_train = data_train


model_embed = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

if os.path.exists(emb_path_train):
    with open(emb_path_train, 'rb') as f:
        emb_train = pkl.load(f)
else:

    emb_train = model_embed.encode(x_train)

    with open(emb_path_train, 'wb') as f:
       pkl.dump(emb_train, f)


if os.path.exists(emb_path_val):
    with open(emb_path_val, 'rb') as f:
        emb_val = pkl.load(f)
else:
    x_val, y_val = data_val
    model_embed = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    emb_val = model_embed.encode(x_train)

    with open(emb_path_train, 'wb') as f:
       pkl.dump(emb_val, f)

# Create the SVM
svm = SVC(random_state=seed)

# Make it an Multilabel classifier
multilabel_classifier = MultiOutputClassifier(svm, n_jobs=-1)

# Fit the data to the Multilabel classifier
multilabel_classifier = multilabel_classifier.fit(emb_train, y_train)

x_val, y_val = data_val

# Get predictions for test data
y_test_pred = multilabel_classifier.predict(emb_val)

# Generate multiclass confusion matrices
matrices = multilabel_confusion_matrix(y_val, y_test_pred)

# Plotting matrices: code
cmd = ConfusionMatrixDisplay(matrices[0], display_labels=np.unique(y_val)).plot()
plt.title('Confusion Matrix for label 1 (type)')
plt.show()
cmd = ConfusionMatrixDisplay(matrices[1], display_labels=np.unique(y_val)).plot()
plt.title('Confusion Matrix for label 2 (color)')
plt.show()