import json
import os
import pandas as pd

works_dir = '/home/k2141255/study_design_classifier/openalex-snapshot/data/works/'

filename_pos_human = '/home/k2141255/study_design_classifier/data/pubmed_articles_pos_human.csv'

filename_neg_human = '/home/k2141255/study_design_classifier/data/pubmed_articles_neg_human.csv'

authors_dict = '/home/k2141255/study_design_classifier/data/authors.json'
open_alex_works_dict = '/home/k2141255/study_design_classifier/data/open_alex_works.json'

print('loading positive data')
pos_data_human = pd.read_csv(
         filename_pos_human,
)
pos_pmids = pos_data_human['pmid']
del pos_data_human

print('loading negative data')
neg_data_human = pd.read_csv(
         filename_neg_human,
)
neg_pmids = neg_data_human['pmid']
del neg_data_human

updated_works_dirs = os.listdir(works_dir)
work_dict = {}

for update_date in updated_works_dirs:
    update_date_dir = os.path.join(works_dir, update_date)
    try:
        works_files = os.listdir(update_date_dir)

        for works_file in works_files:
            works_file_path = os.path.join(update_date_dir, works_file)

            with open(works_file_path, mode='r') as f:
                for line in f:
                    try:
                        json_work = json.loads(line)
                        if 'pmid' in json_work['ids'] and json_work['id'] not in work_dict:
                            work_dict[json_work['id']] = {'authorships': json_work['authorships'], 'title': json_work['title'], 'pmid': json_work['ids']['pmid']}
                            with open(open_alex_works_dict, 'w') as f:
                                json.dump(work_dict, f)
                    except:
                        pass
    except:
        pass
                
        