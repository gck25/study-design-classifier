import os
import sys
import requests
import pickle as pkl
from tqdm import tqdm
import sys
import time
import json
import requests
import itertools
import pandas as pd
from random import choices
from signal import signal, SIGPIPE, SIG_DFL 

from joblib import Parallel, delayed
# from gpuparallel import GPUParallel, delay1d



BASEPMIDOPEN = '../data/pmid_to_openAlex'
PMIDS = '../data/pmids.pkl'
with open(PMIDS, 'rb') as f:
    pmids = pkl.load(f)
    
pmids_total = []

for study_design in pmids:
    for neg_pos in pmids[study_design]:
        for split in pmids[study_design][neg_pos]:
            pmids_total.extend(pmids[study_design][neg_pos][split])
            
            
pos_train_open_alex_ids = []
m = len(pmids_total)//50

# for n in tqdm(range(m)):

pmid_oalex = {}
#     for i in tqdm(range(n)):   
n= int(sys.argv[1] )

pmid_str = '|'.join(map(str, pmids_total[50*n:50*(n+1)]))
save_path = os.path.join(BASEPMIDOPEN, f'pmid_{n}.json')
if not os.path.exists(save_path):
            print('file', n)
            response = requests.get(f"https://api.openalex.org/works?filter=pmid:{pmid_str}&per-page=200")
            while  response.status_code != 200:
                response = requests.get(f"https://api.openalex.org/works?filter=pmid:{pmid_str}&per-page=200")
            for result in response.json()['results']:
                    pmid = int(result['ids']['pmid'].split('/')[-1])
                    pmid_oalex[pmid] = result['id']
            print('saving', n)
            with open(save_path, 'w') as f:
                    json.dump(pmid_oalex, f)
else:
    print('skipping', n)
# def search_for_files(n):
#         pos_train_open_alex_ids = []

#         pmid_oalex = {}
# #     for i in tqdm(range(n)):   
#         pmid_str = '|'.join(map(str, pmids_total[50*n:50*(n+1)]))
#         save_path = os.path.join(BASEPMIDOPEN, f'pmid_{n}.json')
#         while not os.path.exists(save_path):
#             response = requests.get(f"https://api.openalex.org/works?filter=pmid:{pmid_str}&per-page=200")
            
#             if response.status_code != 200:
#                 continue
            
#             for result in response.json()['results']:
#                 pmid = int(result['ids']['pmid'].split('/')[-1])
#                 pmid_oalex[pmid] = result['id']
            
#             with open(save_path, 'w') as f:
#                 json.dump(pmid_oalex, f)
            
            
# n = len(pmids_total)//50
# num_processes =  20
# Parallel(n_jobs=num_processes)(delayed(search_for_files)(i) for i in tqdm(range(n)))