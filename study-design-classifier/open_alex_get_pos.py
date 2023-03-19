import os
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

# from joblib import Parallel, delayed
from gpuparallel import GPUParallel, delayed

num_processes = 100

with open('../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl', 'rb') as f:
    prop_typ_id_dict= pkl.load(f)

pmid_file = 'pos_pmids.json'
if os.path.exists(pmid_file):
    with open(pmid_file, 'r') as f:
        pmids = json.load(f)
else:
    
    pos_ids = []
    neg_ids = []
    for study in prop_typ_id_dict.keys():
        study_dict = prop_typ_id_dict[study]
        train_len = len(study_dict['pos']['train'])
        for pos_neg in study_dict.keys():
            pos_neg_dict = study_dict[pos_neg]

            for split in pos_neg_dict:


                if pos_neg == 'pos':
                    pos_ids.extend(pos_neg_dict[split])
    pos_ids = list(set(pos_ids))

    filename_pos_human = '../data/pubmed_articles_pos_human.csv'
    pos_data_human = pd.read_csv(
             filename_pos_human,
    )
    pmids = list(pos_data_human.iloc[pos_ids]['pmid'])
    
    with open(pmid_file, 'w') as f:
        json.dump(pmids, f)

# with open('neg_pmids.pkl', 'rb') as f:
#     pmids = pkl.load(f)

# def search_for_files(proc_number):
def search_for_files(pmid, worker_id, gpu_id):

#     for pmid in tqdm(pmids[proc_number*len(pmids)//num_processes:(proc_number+1)*len(pmids)//num_processes]):

            open_alex_pmid = f"open_alex_citations/open_alex_pubmed_pmid{pmid}.json"
            
            if not os.path.exists(open_alex_pmid):
                success = False
                retries = 1
                while not success:

                    response = requests.get(f"http://api.openalex.org/works/pmid:{pmid}")
                    if response.status_code == 200:
                        results = response.json()['referenced_works']
                        with open(open_alex_pmid, 'w') as f_pmid:
                             json.dump(results, f_pmid)
                        success = True
#                             cited_by_url = response.json()['cited_by_api_url']


#                             while not success:
#                                 response_cited_by = requests.get(cited_by_url)
#                                 if response_cited_by.status_code == 200:
#                                     results = [res['id'] for res in response_cited_by.json()['results'] ]
#                                     with open(open_alex_pmid, 'w') as f_pmid:
#                                         json.dump(results, f_pmid)

#                                           success = True
#                                 elif response.status_code == 429:
#                                     wait_inner = retries_inner * 30;
#                 #                     wait = 60 * 60 * 24 + 10
#                                     print(f'Error! Waiting %s secs and re-trying...{wait}')
#                                     sys.stdout.flush()
#                                     time.sleep(wait_inner)
#                                     retries_inner += 1
                    elif response.status_code == 429:
                        print(response.status_code)
                        wait = retries * 30;
    #                     wait = 60 * 60 * 24 + 10
                        print(response.content)
                        print(f'Error! Waiting %s secs and re-trying...{wait}')
                        sys.stdout.flush()
                        time.sleep(wait)
                        retries += 1
                    else:
                        break


for pmid in tqdm(pmids):
    search_for_files(pmid, 0, 0)
                        
# success = False
# while not success:
#     try:
#         signal(SIGPIPE,SIG_DFL)                
#         gp = GPUParallel(n_gpu=2, n_workers_per_gpu=2)
#         results = gp(delayed(search_for_files)(pmid) for pmid in pmids)
#         success = True
#     except IOError as e:
#         if e.errno == errno.EPIPE:
#             pass
# Parallel(n_jobs=num_processes)(delayed(search_for_files)(i) for i in range(num_processes))