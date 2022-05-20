import tqdm
import pandas as pd
import tempfile
import csv
import gzip
import xml.etree.cElementTree as ET
import glob
import random
from readers import pmreader
import json
import sys
import os
import math

def abstract_plaintext(abstract):
    out = []
    for sec in abstract:
        if sec["header"] != "_UNSTRUCTURED":
            out.append(sec["header"])
        out.append(sec["text"])

    return "\n".join(out)

def iter_abstracts(filename, num_to_select = None):
    with open(filename, 'rb') as f:
        decompressedFile = gzip.GzipFile(fileobj=f, mode='rb')
        xml_dump = decompressedFile.read()
        xml_parsed = ET.fromstring(xml_dump)
        articles = xml_parsed.findall('PubmedArticle/MedlineCitation')
        if num_to_select:
            article_sample = random.sample(articles, num_to_select)
        else:
            article_sample = articles
        for j, article in enumerate(tqdm.tqdm(article_sample)):
            p_article = pmreader.PubmedCorpusReader(xml_ET=article)
            yield p_article.to_dict()
            
            
def main():
    
    num_to_select = None
    seed = 42

    train_set_size = 8000000
    val_set_size = 600000
    test_non_sys_size = 600000
    test_sys_size = 300000 

    filenames = glob.glob('/data/iain/pubmed/*.gz')
#     all_data = []


    is_train_data = False
    is_val_data = False
    is_test_sys_data = False
    is_test_non_sys_data = False
    
    
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_non_sys_data = pd.DataFrame()
    test_sys_data = pd.DataFrame()


    num_todo = len(filenames)
    
    train_per_file = round(train_set_size/num_todo)
    val_per_file = round(val_set_size/num_todo)
    test_non_sys_per_file = round(test_non_sys_size/num_todo)
    test_sys_size_per_file = round(test_sys_size/num_todo)
    
    for i, fn in tqdm.tqdm(enumerate(filenames)): # this line does *all* >900 files (takes ages)
    # for id in range(len(filenames)):
    # for i, fn in tqdm.tqdm(enumerate([filenames[900]])): # file 900 looks to me to be a decent example; perhaps others better! but some earlier abstracts don't contain much info


            # here we're iterating through *all* files
            # in the final version, probably we'd want to randomly sample, to generate a manageable amount of data
            # I suspect we might not have enough RAM to hold all 300M articles at once, (and so you will need to sample in the loop)
            # but possibly it might work without this, and you could sample from the Pandas dataframe below instead!
            sys.__stdout__.write("file number {}/{} ({}) processing \n".format(i, num_todo, fn)) # optional to see progress from terminal
            print("file number {}/{} ({}) processing \n".format(i, num_todo, fn))
            data = []
            print(fn)
            
            if len(train_data) > train_set_size and len(val_data) > val_set_size and len(test_non_sys_data) > test_non_sys_size and len(test_sys_data) > test_sys_size:
                    break
            
            try:
                for article in iter_abstracts(fn, num_to_select):
                    data.append(article)

                data_pd = pd.DataFrame(data)
                data_pd = data_pd[data_pd['year'] != '']
                data_pd['year'] = list(map(int, data_pd['year']))
                
                
                if len(train_data) < train_set_size:
                    print('entered train')
                    train_subset = data_pd[(data_pd['year']>=1987) & (data_pd['year']<=2014)]
                    if not is_train_data:
                        train_data = train_subset
                        is_train_data = True
                    else:
                        train_data = pd.concat([train_data, train_subset])
                    train_data.to_csv('pubmed_articles_train.csv')
                
                if len(val_data) < val_set_size:
                    print('entered val')
                    val_subset = data_pd[data_pd['year']==2015]
                    if not is_val_data:
                        val_data = val_subset
                        is_val_data = True
                    else:
                        val_data = pd.concat([val_data, val_subset])
                    val_data.to_csv('pubmed_articles_val.csv')
                    
                if len(test_non_sys_data) < test_non_sys_size:
                    print('entered test_non_sys')
                    test_non_sys_subset = data_pd[data_pd['year']==2016]
                    if not is_test_non_sys_data:
                        test_non_sys_data = test_non_sys_subset
                        is_test_non_sys_data = True
                    else:
                        test_non_sys_data = pd.concat([test_non_sys_data, test_non_sys_subset])
                    test_non_sys_data.to_csv('pubmed_articles_test_non_sys.csv')
                    
                if len(test_sys_data) < test_sys_size:
                    print('entered test_sys')
                    test_sys_subset = data_pd[data_pd['year']==2019]
                    if not is_test_sys_data:
                        test_sys_data = test_sys_subset
                        is_test_sys_data = True
                    else:
                        test_sys_data = pd.concat([test_sys_data, test_sys_subset])
                    test_sys_data.to_csv('pubmed_articles_test_sys.csv')
                
            except:
                pass
    
#     article_data = pd.DataFrame(all_data)
#             article_data.to_csv('pubmed_articles.csv')

if __name__ == "__main__":
    main()
