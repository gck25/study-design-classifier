# study-design-classifier


## Classical ML experiments
### Set up environment
```
pip install -r requirements.txt
```
### Training
Use the following notebook to train GBDT:
```
train_GBDT_1gpu_decision_fn.ipynb
```
The output file should be `results_adjusted_metrics_gbdt.json`

Use the following notebook to train the linear classifiers:
```
train_linear_classifier_1gpu_decision_fn.ipynb
```
The output file should be `results_adjusted_metrics_linear_classifiers.json`

## BERT experiments
### Set up environment
```
pip install -r requirements_bert.txt
```
### Training
Run the following command to train the BERT models
```
python train_BERT_1gpu_decision_fn.py
```
The output file should be `results_adjusted_metrics_bert.json`

## Graph-based experiments
### Set up environment
```
pip install -r requirements_gnn.txt
```
### Training

Use the following notebook to train GBDT with BOW and node2vec concatenated:
```
train_bow_node2vec_concat_decision_fn.ipynb
```
The output file should be `results_adjusted_metrics_bow_node2vec_gbdt_concat.json`.

Use the following notebook to train a GBDT on BOW and GBT on node2vec ensemble:
```
train_bow_node2vec_ensemble_decision_fn.py.ipynb
```
The output file should be `results_adjusted_metrics_ensemble_text_node2vec_gbdt.json`

Use the following notebook to train the GNNs:
```
train_gnn_bert_decision_fn.py.ipynb
```
The output file should be `results_adjusted_metrics_gnn_bert.json`

## Results calculation notebooks
### Plot ROC curves
Use the following Jupyter notebook:
```
plot_roc_curves.ipynb
```

### Calculate Wald CI threshold
Use the following notebook to calculate the Wald CI threshold:
```
wald_ci_calculations.ipynb
```

### Select decision function threshold
Use the following notebook to adjust the decision function threshold:
```
select_df_threshold.ipynb
```