import os
import gzip
import json
import string
import pickle as pkl
from tqdm import tqdm
import json


open_alex_pubmed_file = "open_alex_pubmed.json"

works_dir = '/home/k2141255/study_design_classifier/openalex-snapshot/data/works/'

with open('../data/cleaned_pos_neg_train_val_test_dataset_ids.pkl', 'rb') as f:
    prop_typ_id_dict= pkl.load(f)
    
pmids = []
for study in prop_typ_id_dict.keys():
    study_dict = prop_typ_id_dict[study]
    for pos_neg in study_dict.keys():
        pos_neg_dict = study_dict[pos_neg]
        for split in pos_neg_dict:
            pmids.extend(pos_neg_dict[split])
            
pmids = list(set(pmids))

pubmed_works = []

with open('paths_seen.json', 'r') as f_paths:
    paths_seen = json.load(f_paths)

updated_folders = os.listdir(works_dir)
for updated_folder in tqdm(updated_folders):
    updated_dir = os.path.join(works_dir, updated_folder)
    if 'manifest' in updated_dir:
        continue
    parts = os.listdir(updated_dir)
    for part in tqdm(parts):
        part_path = os.path.join(updated_dir, part)
        if part_path in paths_seen:
            continue
        print(part_path)
        with gzip.open(part_path, mode='rb') as f:
            
#             print(g)
            
#             for line in f:
            line = f.readline()
            while line:
                # do stuff with line
                line = f.readline()
                
                try:
                    work = json.loads(line)
                except:
                    print(line)
                if 'pmid' in work['ids'] and int(work['ids']['pmid'].split('/')[-1] in pmids:
                    pubmed_works.append(work)
                    with open(open_alex_pubmed_file, 'w') as f:
                        json.dump(pubmed_works, f)
                        print('entered')
        paths_seen.append(part_path)
        with open('paths_seen.json', 'w') as f_paths:
            json.dump(paths_seen, f_paths)