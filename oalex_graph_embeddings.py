import os
import json
import networkx as nx
from node2vec import Node2Vec

with open('../data/open_alex_graph.json') as f:
    oalex_graph_dict = json.load(f)
    
# Create a graph
graph = nx.Graph(oalex_graph_dict)

temp_folder = '../data/'

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=100)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)


MODEL_DIR = "../models"
EMBEDDING_FILENAME = os.path.join(MODEL_DIR, 'node_model.pkl', temp_folder=temp_folder)
EMBEDDING_MODEL_FILENAME = os.path.join(MODEL_DIR, 'node_embeddings_example.pkl')


model.wv.save_word2vec_format(EMBEDDING_FILENAME)
model.save(EMBEDDING_FILENAME)